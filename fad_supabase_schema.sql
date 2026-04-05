-- ================================================================
--  FETAL ANOMALY DETECTION (FAD) SYSTEM
--  Supabase / PostgreSQL Database Schema
--  Version: 1.0.0
-- ================================================================
--
--  DESIGN PRINCIPLES:
--  1. Supabase Auth (auth.users) handles authentication
--  2. JSONB columns act as extension points for future AI outputs
--  3. RLS (Row Level Security) enforces role-based access
--  4. UUIDs everywhere — no integer IDs leaking
--  5. Triggers auto-create profile + permissions on signup
--  6. Indexes tuned for dashboard queries
--
--  EXTENSION STRATEGY (for when HuggingFace model is deployed):
--  - analysis_results.vlm_output      ← VLM image analysis JSON
--  - analysis_results.llm_output      ← LLM clinical analysis JSON
--  - analysis_results.fusion_output   ← Multimodal fusion JSON
--  - reports.ai_report_json           ← Full structured report JSON
--  - ultrasound_images.preprocessing_metadata ← OpenCV output
--
-- ================================================================


-- ================================================================
-- ENUMS
-- ================================================================

CREATE TYPE user_role         AS ENUM ('doctor', 'admin');
CREATE TYPE severity_level    AS ENUM ('low', 'medium', 'high', 'critical');
CREATE TYPE analysis_status   AS ENUM ('pending', 'in_progress', 'completed', 'failed');
CREATE TYPE alert_status      AS ENUM ('sent', 'acknowledged', 'reviewed');
CREATE TYPE recommendation_type AS ENUM (
  'medication', 'scan', 'consultation', 'procedure', 'follow_up'
);
CREATE TYPE recommendation_priority AS ENUM ('low', 'medium', 'high', 'urgent');
CREATE TYPE anatomical_region AS ENUM ('brain', 'heart', 'nuchal_translucency', 'other');


-- ================================================================
-- UTILITY: auto-update updated_at
-- ================================================================

CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$;


-- ================================================================
-- TABLE 1: PROFILES
-- Extends auth.users — one row per authenticated user
-- ================================================================

CREATE TABLE profiles (
  id                UUID          PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  full_name         VARCHAR(100)  NOT NULL,
  role              user_role     NOT NULL DEFAULT 'doctor',

  -- Doctor-specific details
  specialization    VARCHAR(100),
  hospital_name     VARCHAR(200),
  contact_number    VARCHAR(20),

  -- Account state
  is_active         BOOLEAN       DEFAULT TRUE,

  created_at        TIMESTAMPTZ   DEFAULT NOW(),
  updated_at        TIMESTAMPTZ   DEFAULT NOW()
);

CREATE TRIGGER trg_profiles_updated_at
  BEFORE UPDATE ON profiles
  FOR EACH ROW EXECUTE FUNCTION set_updated_at();


-- ================================================================
-- TABLE 2: PERMISSIONS
-- Per-user feature flags; admin sets these per doctor
-- ================================================================

CREATE TABLE permissions (
  id                    UUID      PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id               UUID      NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,

  can_upload_data       BOOLEAN   DEFAULT TRUE,
  can_view_reports      BOOLEAN   DEFAULT TRUE,
  can_download_reports  BOOLEAN   DEFAULT TRUE,
  can_manage_users      BOOLEAN   DEFAULT FALSE,
  can_delete_data       BOOLEAN   DEFAULT FALSE,

  created_at            TIMESTAMPTZ DEFAULT NOW(),
  updated_at            TIMESTAMPTZ DEFAULT NOW(),

  UNIQUE(user_id)
);

CREATE TRIGGER trg_permissions_updated_at
  BEFORE UPDATE ON permissions
  FOR EACH ROW EXECUTE FUNCTION set_updated_at();


-- ================================================================
-- TABLE 3: PATIENTS
-- ================================================================

CREATE TABLE patients (
  id                      UUID          PRIMARY KEY DEFAULT gen_random_uuid(),

  -- Human-readable identifier shown in UI (e.g., "FAD-2025-0001")
  patient_code            VARCHAR(50)   UNIQUE NOT NULL,

  full_name               VARCHAR(100),
  date_of_birth           DATE,
  contact_number          VARCHAR(20),
  address                 TEXT,

  -- Gestational age at time of registration (weeks)
  gestational_age_weeks   INT,

  -- The doctor who registered this patient
  created_by              UUID          NOT NULL REFERENCES profiles(id),

  created_at              TIMESTAMPTZ   DEFAULT NOW(),
  updated_at              TIMESTAMPTZ   DEFAULT NOW()
);

CREATE TRIGGER trg_patients_updated_at
  BEFORE UPDATE ON patients
  FOR EACH ROW EXECUTE FUNCTION set_updated_at();


-- ================================================================
-- TABLE 4: CLINICAL DATA
-- Maternal/clinical information entered or uploaded by doctor
-- ================================================================

CREATE TABLE clinical_data (
  id                      UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id              UUID          NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
  uploaded_by             UUID          NOT NULL REFERENCES profiles(id),

  -- Core vitals
  maternal_age            INT,
  gestational_age_weeks   INT,
  weight_kg               DECIMAL(5,2),
  height_cm               DECIMAL(5,2),
  blood_pressure          VARCHAR(20),   -- e.g. "120/80"
  heart_rate              INT,

  -- Medical history (free text — structured input later)
  medical_history         TEXT,
  previous_pregnancies    INT           DEFAULT 0,
  previous_complications  TEXT,
  current_medications     TEXT,
  genetic_risk_factors    TEXT,

  -- File upload support (CSV/PDF clinical file)
  file_name               VARCHAR(255),
  storage_path            VARCHAR(500),  -- Supabase Storage bucket path

  -- ⬇ EXTENSION POINT: any extra parsed fields from the file or future form fields
  extra_fields            JSONB         DEFAULT '{}',

  created_at              TIMESTAMPTZ   DEFAULT NOW(),
  updated_at              TIMESTAMPTZ   DEFAULT NOW()
);

CREATE TRIGGER trg_clinical_data_updated_at
  BEFORE UPDATE ON clinical_data
  FOR EACH ROW EXECUTE FUNCTION set_updated_at();


-- ================================================================
-- TABLE 5: ULTRASOUND IMAGES
-- ================================================================

CREATE TABLE ultrasound_images (
  id                            UUID              PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id                    UUID              NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
  uploaded_by                   UUID              NOT NULL REFERENCES profiles(id),

  -- Supabase Storage
  file_name                     VARCHAR(255)      NOT NULL,
  storage_path                  VARCHAR(500)      NOT NULL,  -- path inside bucket
  file_size_kb                  INT,
  image_format                  VARCHAR(10),       -- 'png', 'jpg', 'dicom'

  -- Clinical metadata
  anatomical_region             anatomical_region,
  anatomical_plane              VARCHAR(100),      -- e.g. 'transventricular', 'four_chamber_view'
  scan_date                     DATE,
  gestational_age_at_scan_weeks INT,

  -- Processing state
  is_processed                  BOOLEAN           DEFAULT FALSE,

  -- ⬇ EXTENSION POINT: OpenCV preprocessing results, quality scores, etc.
  preprocessing_metadata        JSONB             DEFAULT '{}',

  created_at                    TIMESTAMPTZ       DEFAULT NOW()
);


-- ================================================================
-- TABLE 6: ANALYSIS RESULTS
-- One row per analysis run; JSONB columns absorb AI outputs later
-- ================================================================

CREATE TABLE analysis_results (
  id                      UUID              PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id              UUID              NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
  image_id                UUID              REFERENCES ultrasound_images(id) ON DELETE SET NULL,
  clinical_data_id        UUID              REFERENCES clinical_data(id) ON DELETE SET NULL,

  -- Lifecycle
  status                  analysis_status   DEFAULT 'pending',
  ai_model_version        VARCHAR(50),
  processing_time_ms      INT,

  -- ⬇ EXTENSION POINTS — populate once HuggingFace model is live
  --    vlm_output    : raw JSON from the Vision Language Model
  --    llm_output    : raw JSON from the LLM clinical analyzer
  --    fusion_output : combined multimodal JSON
  vlm_output              JSONB             DEFAULT '{}',
  llm_output              JSONB             DEFAULT '{}',
  fusion_output           JSONB             DEFAULT '{}',

  -- Aggregate scores (derived from fusion_output or set manually for now)
  overall_confidence_score DECIMAL(5,4),    -- 0.0000 – 1.0000
  overall_risk_level       severity_level,

  -- Error tracking
  error_message           TEXT,

  created_at              TIMESTAMPTZ       DEFAULT NOW(),
  updated_at              TIMESTAMPTZ       DEFAULT NOW()
);

CREATE TRIGGER trg_analysis_updated_at
  BEFORE UPDATE ON analysis_results
  FOR EACH ROW EXECUTE FUNCTION set_updated_at();


-- ================================================================
-- TABLE 7: DETECTED ANOMALIES
-- Child of analysis_results; one row per detected anomaly
-- ================================================================

CREATE TABLE detected_anomalies (
  id                UUID              PRIMARY KEY DEFAULT gen_random_uuid(),
  analysis_id       UUID              NOT NULL REFERENCES analysis_results(id) ON DELETE CASCADE,

  anomaly_name      VARCHAR(100)      NOT NULL,
  anomaly_region    anatomical_region DEFAULT 'other',
  description       TEXT,

  severity_level    severity_level    NOT NULL,
  confidence_score  DECIMAL(5,4),     -- 0.0000 – 1.0000

  -- Spatial location (for heatmap overlay on image)
  location_label    VARCHAR(100),     -- human-readable e.g. "posterior fossa"
  bounding_box      JSONB,            -- { "x": 0, "y": 0, "width": 100, "height": 80 }

  -- Explainability (XAI)
  explanation       TEXT,
  heatmap_path      VARCHAR(500),     -- Supabase Storage path for Grad-CAM heatmap

  is_critical       BOOLEAN           DEFAULT FALSE,
  detected_at       TIMESTAMPTZ       DEFAULT NOW()
);


-- ================================================================
-- TABLE 8: RISK FACTORS
-- Child of analysis_results; one row per identified risk factor
-- ================================================================

CREATE TABLE risk_factors (
  id                    UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
  analysis_id           UUID            NOT NULL REFERENCES analysis_results(id) ON DELETE CASCADE,

  risk_factor_name      VARCHAR(100)    NOT NULL,
  -- Category: 'maternal_age', 'genetic', 'imaging_finding', 'clinical_history'
  risk_category         VARCHAR(60),
  risk_level            severity_level  NOT NULL,
  description           TEXT,
  contributing_factors  TEXT,

  created_at            TIMESTAMPTZ     DEFAULT NOW()
);


-- ================================================================
-- TABLE 9: REPORTS
-- Final deliverable per analysis; PDF stored in Supabase Storage
-- ================================================================

CREATE TABLE reports (
  id              UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id      UUID          NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
  analysis_id     UUID          NOT NULL REFERENCES analysis_results(id) ON DELETE CASCADE,
  generated_by    UUID          NOT NULL REFERENCES profiles(id),

  title           VARCHAR(255),
  summary         TEXT,

  -- Supabase Storage path to the generated PDF
  pdf_storage_path VARCHAR(500),

  is_finalized    BOOLEAN       DEFAULT FALSE,
  finalized_at    TIMESTAMPTZ,

  -- ⬇ EXTENSION POINT: Full structured AI report JSON from model
  --    Populate this after HuggingFace model is deployed.
  --    Structure will be: { findings: [], recommendations: [], summary: "", ... }
  ai_report_json  JSONB         DEFAULT '{}',

  created_at      TIMESTAMPTZ   DEFAULT NOW(),
  updated_at      TIMESTAMPTZ   DEFAULT NOW()
);

CREATE TRIGGER trg_reports_updated_at
  BEFORE UPDATE ON reports
  FOR EACH ROW EXECUTE FUNCTION set_updated_at();


-- ================================================================
-- TABLE 10: TREATMENT RECOMMENDATIONS
-- Child of reports; AI-generated follow-up actions
-- ================================================================

CREATE TABLE treatment_recommendations (
  id                       UUID                     PRIMARY KEY DEFAULT gen_random_uuid(),
  report_id                UUID                     NOT NULL REFERENCES reports(id) ON DELETE CASCADE,

  recommendation_type      recommendation_type      NOT NULL,
  recommendation_text      TEXT                     NOT NULL,
  priority                 recommendation_priority  NOT NULL,

  specialist_type          VARCHAR(100),             -- e.g. "pediatric cardiologist"
  recommended_timeframe    VARCHAR(100),             -- e.g. "within 48 hours"
  notes                    TEXT,

  -- Track whether this was AI-generated or doctor-entered
  ai_generated             BOOLEAN                  DEFAULT TRUE,

  created_at               TIMESTAMPTZ              DEFAULT NOW()
);


-- ================================================================
-- TABLE 11: ALERTS
-- Auto-generated when critical anomalies are detected
-- ================================================================

CREATE TABLE alerts (
  id              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id      UUID            NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
  analysis_id     UUID            NOT NULL REFERENCES analysis_results(id) ON DELETE CASCADE,
  anomaly_id      UUID            REFERENCES detected_anomalies(id) ON DELETE SET NULL,
  doctor_id       UUID            NOT NULL REFERENCES profiles(id),

  alert_message   TEXT            NOT NULL,
  severity        severity_level  NOT NULL,
  status          alert_status    DEFAULT 'sent',

  acknowledged_at TIMESTAMPTZ,
  reviewed_at     TIMESTAMPTZ,

  created_at      TIMESTAMPTZ     DEFAULT NOW()
);


-- ================================================================
-- TABLE 12: DOWNLOAD LOGS
-- Audit trail for every report download
-- ================================================================

CREATE TABLE download_logs (
  id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  report_id       UUID        NOT NULL REFERENCES reports(id) ON DELETE CASCADE,
  downloaded_by   UUID        NOT NULL REFERENCES profiles(id),
  ip_address      VARCHAR(45),
  created_at      TIMESTAMPTZ DEFAULT NOW()
);


-- ================================================================
-- TABLE 13: ACTIVITY LOGS
-- General audit trail for user actions
-- ================================================================

CREATE TABLE activity_logs (
  id                   UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id              UUID        NOT NULL REFERENCES profiles(id),

  -- e.g. 'login', 'logout', 'upload_image', 'upload_clinical',
  --      'view_report', 'download_report', 'acknowledge_alert',
  --      'register_doctor', 'remove_doctor'
  activity_type        VARCHAR(60) NOT NULL,
  activity_description TEXT,

  -- Optional link to a patient record
  patient_id           UUID        REFERENCES patients(id) ON DELETE SET NULL,

  -- ⬇ Any extra structured data for this activity (request body, params, etc.)
  metadata             JSONB       DEFAULT '{}',
  ip_address           VARCHAR(45),

  created_at           TIMESTAMPTZ DEFAULT NOW()
);


-- ================================================================
-- TABLE 14: SYSTEM CONFIGURATION
-- Admin-managed key-value store; JSONB values for flexibility
-- ================================================================

CREATE TABLE system_config (
  id            UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  config_key    VARCHAR(100) UNIQUE NOT NULL,
  config_value  JSONB,       -- supports strings, booleans, numbers, objects
  description   TEXT,
  updated_by    UUID        REFERENCES profiles(id),
  updated_at    TIMESTAMPTZ DEFAULT NOW()
);


-- ================================================================
-- INDEXES
-- ================================================================

-- Patients
CREATE INDEX idx_patients_created_by    ON patients(created_by);
CREATE INDEX idx_patients_code          ON patients(patient_code);

-- Clinical Data
CREATE INDEX idx_clinical_patient       ON clinical_data(patient_id);

-- Ultrasound Images
CREATE INDEX idx_ultrasound_patient     ON ultrasound_images(patient_id);
CREATE INDEX idx_ultrasound_region      ON ultrasound_images(anatomical_region);
CREATE INDEX idx_ultrasound_processed   ON ultrasound_images(is_processed);

-- Analysis Results
CREATE INDEX idx_analysis_patient       ON analysis_results(patient_id);
CREATE INDEX idx_analysis_status        ON analysis_results(status);
CREATE INDEX idx_analysis_risk          ON analysis_results(overall_risk_level);

-- Detected Anomalies
CREATE INDEX idx_anomalies_analysis     ON detected_anomalies(analysis_id);
CREATE INDEX idx_anomalies_critical     ON detected_anomalies(is_critical);
CREATE INDEX idx_anomalies_severity     ON detected_anomalies(severity_level);

-- Risk Factors
CREATE INDEX idx_risk_analysis          ON risk_factors(analysis_id);

-- Reports
CREATE INDEX idx_reports_patient        ON reports(patient_id);
CREATE INDEX idx_reports_finalized      ON reports(is_finalized);
CREATE INDEX idx_reports_generated_by   ON reports(generated_by);

-- Alerts
CREATE INDEX idx_alerts_doctor          ON alerts(doctor_id);
CREATE INDEX idx_alerts_status          ON alerts(status);
CREATE INDEX idx_alerts_severity        ON alerts(severity);
CREATE INDEX idx_alerts_patient         ON alerts(patient_id);

-- Activity Logs
CREATE INDEX idx_activity_user          ON activity_logs(user_id);
CREATE INDEX idx_activity_type          ON activity_logs(activity_type);
CREATE INDEX idx_activity_timestamp     ON activity_logs(created_at DESC);


-- ================================================================
-- AUTO-PROVISION: New user signup → profile + permissions
-- ================================================================

CREATE OR REPLACE FUNCTION handle_new_user()
RETURNS TRIGGER LANGUAGE plpgsql SECURITY DEFINER AS $$
BEGIN
  -- Create profile row
  INSERT INTO profiles (id, full_name, role)
  VALUES (
    NEW.id,
    COALESCE(NEW.raw_user_meta_data->>'full_name', NEW.email),
    COALESCE((NEW.raw_user_meta_data->>'role')::user_role, 'doctor')
  )
  ON CONFLICT (id) DO NOTHING;

  -- Create default permissions row
  INSERT INTO permissions (user_id)
  VALUES (NEW.id)
  ON CONFLICT (user_id) DO NOTHING;

  RETURN NEW;
END;
$$;

CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION handle_new_user();


-- ================================================================
-- HELPER FUNCTION: get current user's role (for RLS policies)
-- ================================================================

CREATE OR REPLACE FUNCTION get_my_role()
RETURNS user_role LANGUAGE sql STABLE SECURITY DEFINER AS $$
  SELECT role FROM profiles WHERE id = auth.uid();
$$;


-- ================================================================
-- ROW LEVEL SECURITY (RLS)
-- ================================================================

ALTER TABLE profiles                 ENABLE ROW LEVEL SECURITY;
ALTER TABLE permissions              ENABLE ROW LEVEL SECURITY;
ALTER TABLE patients                 ENABLE ROW LEVEL SECURITY;
ALTER TABLE clinical_data            ENABLE ROW LEVEL SECURITY;
ALTER TABLE ultrasound_images        ENABLE ROW LEVEL SECURITY;
ALTER TABLE analysis_results         ENABLE ROW LEVEL SECURITY;
ALTER TABLE detected_anomalies       ENABLE ROW LEVEL SECURITY;
ALTER TABLE risk_factors             ENABLE ROW LEVEL SECURITY;
ALTER TABLE reports                  ENABLE ROW LEVEL SECURITY;
ALTER TABLE treatment_recommendations ENABLE ROW LEVEL SECURITY;
ALTER TABLE alerts                   ENABLE ROW LEVEL SECURITY;
ALTER TABLE download_logs            ENABLE ROW LEVEL SECURITY;
ALTER TABLE activity_logs            ENABLE ROW LEVEL SECURITY;
ALTER TABLE system_config            ENABLE ROW LEVEL SECURITY;


-- ── PROFILES ──────────────────────────────────────────────────
CREATE POLICY "Own profile: read"      ON profiles FOR SELECT USING (auth.uid() = id);
CREATE POLICY "Admin: read all"        ON profiles FOR SELECT USING (get_my_role() = 'admin');
CREATE POLICY "Own profile: update"    ON profiles FOR UPDATE USING (auth.uid() = id);
CREATE POLICY "Admin: manage all"      ON profiles FOR ALL    USING (get_my_role() = 'admin');

-- ── PERMISSIONS ────────────────────────────────────────────────
CREATE POLICY "Admin: manage all"      ON permissions FOR ALL    USING (get_my_role() = 'admin');
CREATE POLICY "Own perms: read"        ON permissions FOR SELECT USING (user_id = auth.uid());

-- ── PATIENTS ───────────────────────────────────────────────────
CREATE POLICY "Doctor: own patients"   ON patients FOR SELECT USING (created_by = auth.uid());
CREATE POLICY "Admin: all patients"    ON patients FOR SELECT USING (get_my_role() = 'admin');
CREATE POLICY "Doctor: insert"         ON patients FOR INSERT  WITH CHECK (auth.uid() = created_by);
CREATE POLICY "Doctor: update own"     ON patients FOR UPDATE  USING (created_by = auth.uid());

-- ── CLINICAL DATA ──────────────────────────────────────────────
CREATE POLICY "Doctor: own uploads"    ON clinical_data FOR SELECT USING (uploaded_by = auth.uid());
CREATE POLICY "Admin: all"             ON clinical_data FOR SELECT USING (get_my_role() = 'admin');
CREATE POLICY "Doctor: insert"         ON clinical_data FOR INSERT  WITH CHECK (auth.uid() = uploaded_by);

-- ── ULTRASOUND IMAGES ──────────────────────────────────────────
CREATE POLICY "Doctor: own images"     ON ultrasound_images FOR SELECT USING (uploaded_by = auth.uid());
CREATE POLICY "Admin: all images"      ON ultrasound_images FOR SELECT USING (get_my_role() = 'admin');
CREATE POLICY "Doctor: insert"         ON ultrasound_images FOR INSERT  WITH CHECK (auth.uid() = uploaded_by);
CREATE POLICY "Doctor: update own"     ON ultrasound_images FOR UPDATE  USING (uploaded_by = auth.uid());

-- ── ANALYSIS RESULTS ───────────────────────────────────────────
CREATE POLICY "Doctor: own patient analysis" ON analysis_results FOR SELECT
  USING (patient_id IN (SELECT id FROM patients WHERE created_by = auth.uid()));
CREATE POLICY "Admin: all analysis"          ON analysis_results FOR SELECT
  USING (get_my_role() = 'admin');
CREATE POLICY "Doctor: insert"               ON analysis_results FOR INSERT
  WITH CHECK (patient_id IN (SELECT id FROM patients WHERE created_by = auth.uid()));
CREATE POLICY "Doctor: update own"           ON analysis_results FOR UPDATE
  USING (patient_id IN (SELECT id FROM patients WHERE created_by = auth.uid()));

-- ── DETECTED ANOMALIES ─────────────────────────────────────────
CREATE POLICY "Doctor: own anomalies" ON detected_anomalies FOR SELECT
  USING (analysis_id IN (
    SELECT a.id FROM analysis_results a
    JOIN patients p ON p.id = a.patient_id
    WHERE p.created_by = auth.uid()
  ));
CREATE POLICY "Admin: all anomalies"  ON detected_anomalies FOR SELECT USING (get_my_role() = 'admin');
CREATE POLICY "Insert anomalies"      ON detected_anomalies FOR INSERT WITH CHECK (TRUE);

-- ── RISK FACTORS ───────────────────────────────────────────────
CREATE POLICY "Doctor: own risk factors" ON risk_factors FOR SELECT
  USING (analysis_id IN (
    SELECT a.id FROM analysis_results a
    JOIN patients p ON p.id = a.patient_id
    WHERE p.created_by = auth.uid()
  ));
CREATE POLICY "Admin: all risk factors" ON risk_factors FOR SELECT USING (get_my_role() = 'admin');
CREATE POLICY "Insert risk factors"     ON risk_factors FOR INSERT WITH CHECK (TRUE);

-- ── REPORTS ────────────────────────────────────────────────────
CREATE POLICY "Doctor: own reports"  ON reports FOR SELECT USING (generated_by = auth.uid());
CREATE POLICY "Admin: all reports"   ON reports FOR SELECT USING (get_my_role() = 'admin');
CREATE POLICY "Doctor: insert"       ON reports FOR INSERT  WITH CHECK (auth.uid() = generated_by);
CREATE POLICY "Doctor: update own"   ON reports FOR UPDATE  USING (generated_by = auth.uid());

-- ── TREATMENT RECOMMENDATIONS ──────────────────────────────────
CREATE POLICY "Doctor: own recommendations" ON treatment_recommendations FOR SELECT
  USING (report_id IN (SELECT id FROM reports WHERE generated_by = auth.uid()));
CREATE POLICY "Admin: all"                  ON treatment_recommendations FOR SELECT
  USING (get_my_role() = 'admin');
CREATE POLICY "Insert recommendations"      ON treatment_recommendations FOR INSERT WITH CHECK (TRUE);

-- ── ALERTS ─────────────────────────────────────────────────────
CREATE POLICY "Doctor: own alerts"     ON alerts FOR SELECT USING (doctor_id = auth.uid());
CREATE POLICY "Admin: all alerts"      ON alerts FOR SELECT USING (get_my_role() = 'admin');
CREATE POLICY "Doctor: acknowledge"    ON alerts FOR UPDATE  USING (doctor_id = auth.uid());
CREATE POLICY "Insert alerts"          ON alerts FOR INSERT  WITH CHECK (TRUE);

-- ── DOWNLOAD LOGS ──────────────────────────────────────────────
CREATE POLICY "Own downloads"          ON download_logs FOR SELECT USING (downloaded_by = auth.uid());
CREATE POLICY "Admin: all downloads"   ON download_logs FOR SELECT USING (get_my_role() = 'admin');
CREATE POLICY "Doctor: insert"         ON download_logs FOR INSERT  WITH CHECK (auth.uid() = downloaded_by);

-- ── ACTIVITY LOGS ──────────────────────────────────────────────
CREATE POLICY "Own activity"           ON activity_logs FOR SELECT USING (user_id = auth.uid());
CREATE POLICY "Admin: all activity"    ON activity_logs FOR SELECT USING (get_my_role() = 'admin');
CREATE POLICY "Insert activity"        ON activity_logs FOR INSERT  WITH CHECK (auth.uid() = user_id);

-- ── SYSTEM CONFIG ──────────────────────────────────────────────
CREATE POLICY "Admin: full control"    ON system_config FOR ALL    USING (get_my_role() = 'admin');
CREATE POLICY "Any user: read"         ON system_config FOR SELECT USING (auth.uid() IS NOT NULL);


-- ================================================================
-- SEED: Default system configuration values
-- ================================================================

INSERT INTO system_config (config_key, config_value, description) VALUES
  ('ai_model_enabled',        'false',
   'Toggle AI model inference on/off. Set to true after HuggingFace deployment.'),

  ('huggingface_model_url',   '""',
   'HuggingFace Inference API endpoint for the VLM model.'),

  ('llm_model_url',           '""',
   'HuggingFace or OpenAI endpoint for the LLM clinical analyzer.'),

  ('max_image_size_mb',       '10',
   'Maximum file size (MB) allowed for ultrasound image uploads.'),

  ('allowed_image_formats',   '["png", "jpg", "jpeg", "dicom"]',
   'Accepted file formats for ultrasound image uploads.'),

  ('alert_auto_send',         'true',
   'If true, alerts are sent automatically when critical anomalies are detected.'),

  ('report_retention_days',   '365',
   'How many days to retain finalized reports in storage.'),

  ('session_timeout_minutes', '15',
   'Idle session timeout before automatic logout.');


-- ================================================================
-- DASHBOARD QUERY HELPERS (views)
-- These power the main dashboard cards and tables
-- ================================================================

-- View: per-doctor summary stats
CREATE OR REPLACE VIEW v_doctor_dashboard AS
SELECT
  p.id                                      AS doctor_id,
  p.full_name                               AS doctor_name,
  COUNT(DISTINCT pt.id)                     AS total_patients,
  COUNT(DISTINCT r.id)                      AS total_reports,
  COUNT(DISTINCT CASE WHEN al.status = 'sent' THEN al.id END) AS unread_alerts,
  COUNT(DISTINCT CASE WHEN ar.status = 'completed' THEN ar.id END) AS completed_analyses,
  COUNT(DISTINCT CASE WHEN ar.status = 'pending'   THEN ar.id END) AS pending_analyses
FROM profiles p
LEFT JOIN patients pt           ON pt.created_by        = p.id
LEFT JOIN analysis_results ar   ON ar.patient_id         = pt.id
LEFT JOIN reports r             ON r.patient_id          = pt.id AND r.generated_by = p.id
LEFT JOIN alerts al             ON al.doctor_id          = p.id
WHERE p.role = 'doctor'
GROUP BY p.id, p.full_name;


-- View: admin overview
CREATE OR REPLACE VIEW v_admin_overview AS
SELECT
  (SELECT COUNT(*) FROM profiles WHERE role = 'doctor') AS total_doctors,
  (SELECT COUNT(*) FROM patients)                        AS total_patients,
  (SELECT COUNT(*) FROM analysis_results)                AS total_analyses,
  (SELECT COUNT(*) FROM reports WHERE is_finalized)      AS finalized_reports,
  (SELECT COUNT(*) FROM alerts WHERE status = 'sent')    AS pending_alerts,
  (SELECT COUNT(*) FROM alerts WHERE severity = 'critical' AND status = 'sent') AS critical_unread;


-- View: recent patients with latest analysis status (doctor-scoped by RLS)
CREATE OR REPLACE VIEW v_recent_patients AS
SELECT
  pt.id,
  pt.patient_code,
  pt.full_name,
  pt.gestational_age_weeks,
  pt.created_at,
  ar.status                     AS latest_analysis_status,
  ar.overall_risk_level         AS risk_level,
  ar.overall_confidence_score   AS confidence,
  r.id                          AS latest_report_id,
  r.is_finalized                AS report_finalized
FROM patients pt
LEFT JOIN LATERAL (
  SELECT * FROM analysis_results
  WHERE patient_id = pt.id
  ORDER BY created_at DESC
  LIMIT 1
) ar ON TRUE
LEFT JOIN LATERAL (
  SELECT * FROM reports
  WHERE patient_id = pt.id
  ORDER BY created_at DESC
  LIMIT 1
) r ON TRUE
ORDER BY pt.created_at DESC;


-- View: unread alerts with patient and anomaly context
CREATE OR REPLACE VIEW v_active_alerts AS
SELECT
  al.id,
  al.severity,
  al.alert_message,
  al.status,
  al.created_at,
  pt.patient_code,
  pt.full_name      AS patient_name,
  da.anomaly_name,
  da.anomaly_region,
  al.doctor_id
FROM alerts al
JOIN patients pt           ON pt.id = al.patient_id
LEFT JOIN detected_anomalies da ON da.id = al.anomaly_id
WHERE al.status != 'reviewed'
ORDER BY
  CASE al.severity
    WHEN 'critical' THEN 1
    WHEN 'high'     THEN 2
    WHEN 'medium'   THEN 3
    ELSE                 4
  END,
  al.created_at DESC;


-- ================================================================
-- END OF SCHEMA
-- ================================================================
--
--  NEXT STEPS (when AI model is ready):
--
--  1. Upload image → Supabase Storage bucket "ultrasound-images"
--  2. Call HuggingFace VLM endpoint with image URL
--  3. Call HuggingFace/OpenAI LLM endpoint with clinical data
--  4. INSERT into analysis_results with vlm_output, llm_output, fusion_output
--  5. Parse JSON → INSERT rows into detected_anomalies + risk_factors
--  6. INSERT into reports + treatment_recommendations
--  7. If any anomaly.is_critical → INSERT into alerts
--
--  All existing UI pages need zero schema changes for this flow.
--
-- ================================================================
