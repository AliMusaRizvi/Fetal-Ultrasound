import React, { useState } from 'react';
import { motion } from 'motion/react';
import {
  UploadCloud, File, X, CheckCircle2, AlertCircle,
  BrainCircuit, Activity, FileText, HeartPulse, TestTube,
  Download, RotateCcw
} from 'lucide-react';
import { runVLMPipeline, type VLMPipelineResult } from '../../lib/vlm';
import { runMedMOAnalysis, buildPatientText, type MedMOReport, type ClinicalFormData } from '../../lib/llm';
import { supabase } from '../../lib/supabase';

interface AnalysisState {
  step: 'idle' | 'uploading' | 'vlm' | 'llm' | 'saving' | 'done' | 'error';
  progress: number;
  error?: string;
}

interface Results {
  vlm: VLMPipelineResult;
  llm: MedMOReport;
  analysisId?: string;
  reportCode?: string;
}

export default function UploadScan() {
  const [dragActive, setDragActive] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [state, setState] = useState<AnalysisState>({ step: 'idle', progress: 0 });
  const [results, setResults] = useState<Results | null>(null);

  const [clinicalData, setClinicalData] = useState<ClinicalFormData>({
    age: '',
    bmi: '',
    blood_pressure: '',
    gestational_age: '',
    previous_c_section: 'No',
    previous_miscarriages: '0',
    previous_preterm_birth: 'No',
    chronic_hypertension: 'No',
    diabetes: 'No',
    gestational_diabetes: 'No',
    preeclampsia_history: 'No',
    multiple_pregnancy: 'No',
    smoking: 'No',
    alcohol_use: 'No',
    family_history: '',
    hb_level: '',
    urine_protein: '',
    blood_glucose: '',
    risk_level: 'Low',
  });

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target;
    setClinicalData(prev => ({ ...prev, [name]: value }));
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(e.type === 'dragenter' || e.type === 'dragover');
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files?.[0]) setFile(e.dataTransfer.files[0]);
  };

  // ─── Main analysis function ──────────────────────────────────────────────────
  const handleAnalyze = async () => {
    if (!file) return;
    setState({ step: 'uploading', progress: 10 });
    setResults(null);

    try {
      const userId = localStorage.getItem('userId') || '';

      // ── Step 1: Upload image to Supabase Storage ────────────────────────────
      const storagePath = `scans/${userId}/${Date.now()}_${file.name}`;
      const { error: storageError } = await supabase.storage
        .from('ultrasound_images')
        .upload(storagePath, file, { upsert: false });

      if (storageError && !storageError.message.includes('not found')) {
        console.warn('Storage upload failed (bucket may not exist):', storageError.message);
      }

      setState({ step: 'vlm', progress: 30 });

      // ── Step 2: Run VLM analysis ─────────────────────────────────────────────
      const vlmResult = await runVLMPipeline(file);

      setState({ step: 'llm', progress: 60 });

      // ── Step 3: Run LLM analysis in parallel ─────────────────────────────────
      const llmResult = await runMedMOAnalysis(clinicalData);

      setState({ step: 'saving', progress: 80 });

      // ── Step 4: Save to database ─────────────────────────────────────────────
      const analysisId = await saveToDatabase(vlmResult, llmResult, storagePath, userId);

      setState({ step: 'done', progress: 100 });
      setResults({ vlm: vlmResult, llm: llmResult, analysisId: analysisId ?? undefined });

    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Analysis failed. Please try again.';
      setState({ step: 'error', progress: 0, error: msg });
    }
  };

  // ─── Save everything to Supabase ────────────────────────────────────────────
  const saveToDatabase = async (
    vlm: VLMPipelineResult,
    llm: MedMOReport,
    storagePath: string,
    userId: string
  ): Promise<string | null> => {
    try {
      const doctorId = userId || null;

      // 1. Create patient (anonymous for now — future improvement: patient registration)
      const { data: patient } = await supabase
        .from('patients')
        .insert({
          gestational_age_weeks: clinicalData.gestational_age ? parseInt(clinicalData.gestational_age) : null,
          created_by: doctorId,
        })
        .select('id, patient_code')
        .single();

      const patientId = patient?.id || null;

      // 2. Save clinical data — all form fields
      const { data: clinical } = await supabase
        .from('clinical_data')
        .insert({
          patient_id: patientId,
          doctor_id: doctorId,
          maternal_age: clinicalData.age ? parseInt(clinicalData.age) : null,
          bmi: clinicalData.bmi ? parseFloat(clinicalData.bmi) : null,
          blood_pressure: clinicalData.blood_pressure || null,
          gestational_age_weeks: clinicalData.gestational_age ? parseInt(clinicalData.gestational_age) : null,
          previous_c_section: clinicalData.previous_c_section,
          previous_miscarriages: parseInt(clinicalData.previous_miscarriages) || 0,
          previous_preterm_birth: clinicalData.previous_preterm_birth,
          multiple_pregnancy: clinicalData.multiple_pregnancy,
          chronic_hypertension: clinicalData.chronic_hypertension,
          diabetes: clinicalData.diabetes,
          gestational_diabetes: clinicalData.gestational_diabetes,
          preeclampsia_history: clinicalData.preeclampsia_history,
          smoking: clinicalData.smoking,
          alcohol_use: clinicalData.alcohol_use,
          family_history: clinicalData.family_history || null,
          hb_level: clinicalData.hb_level ? parseFloat(clinicalData.hb_level) : null,
          urine_protein: clinicalData.urine_protein || null,
          blood_glucose: clinicalData.blood_glucose ? parseFloat(clinicalData.blood_glucose) : null,
          assessed_risk_level: clinicalData.risk_level as string,
          llm_prompt_text: buildPatientText(clinicalData),
        })
        .select('id')
        .single();

      const clinicalId = clinical?.id || null;

      // 3. Save image record
      const { data: imageRecord } = await supabase
        .from('ultrasound_images')
        .insert({
          patient_id: patientId,
          clinical_data_id: clinicalId,
          doctor_id: doctorId,
          file_name: file!.name,
          storage_path: storagePath,
          file_size_kb: file!.size / 1024,
          image_format: file!.type.split('/')[1]?.toUpperCase() || 'JPEG',
          anatomical_plane: vlm.plane.label,
          is_processed: true,
        })
        .select('id')
        .single();

      const imageId = imageRecord?.id || null;

      // 4. Determine overall risk (combine VLM + LLM)
      const llmRisk = llm.risk_level;
      let vlmHasCritical = false;
      if (vlm.nt_markers && vlm.nt_markers.risk === 'High') vlmHasCritical = true;
      if (vlm.heart_segmentation && vlm.heart_segmentation.ctr > 0.55) vlmHasCritical = true;
      if (vlm.brain_anomaly && !vlm.brain_anomaly.is_normal) vlmHasCritical = true;

      const overallRisk = vlmHasCritical && llmRisk === 'Low' ? 'Moderate' :
        vlmHasCritical && llmRisk === 'High' ? 'High' : llmRisk;

      // Build summary text
      const summary = buildSummary(vlm, llm);

      // 5. Save analysis result
      const { data: analysis } = await supabase
        .from('analysis_results')
        .insert({
          patient_id: patientId,
          image_id: imageId,
          clinical_data_id: clinicalId,
          doctor_id: doctorId,
          analysis_type: 'Combined',
          status: 'completed',
          vlm_output: {
            plane: vlm.plane,
            brain_anomaly: vlm.brain_anomaly || null,
            nt_markers: vlm.nt_markers || null,
            heart_segmentation: vlm.heart_segmentation || null,
            routed_to: vlm.routed_to,
          },
          llm_output: {
            risk_level: llm.risk_level,
            sections: llm.sections,
            risk_factors: llm.risk_factors,
            raw_response: llm.raw_response,
          },
          fusion_output: {
            overall_risk: overallRisk,
            vlm_critical: vlmHasCritical,
            combined_summary: summary,
            requires_review: vlmHasCritical || llmRisk === 'High',
          },
          overall_risk_level: overallRisk,
          summary,
        })
        .select('id')
        .single();

      const analysisId = analysis?.id || null;

      // 6. Save detected anomalies
      await saveAnomalies(vlm, analysisId, patientId);

      // 7. Save report with MedMO structured JSON
      const { data: report } = await supabase
        .from('reports')
        .insert({
          patient_id: patientId,
          analysis_id: analysisId,
          doctor_id: doctorId,
          title: `Fetal Ultrasound Analysis — ${patient?.patient_code || 'Unknown Patient'}`,
          summary,
          findings_label: buildFindingsLabel(vlm, llm),
          medmo_report_json: llm.sections,
          overall_risk: overallRisk,
          is_finalized: false,
        })
        .select('id, report_code')
        .single();

      // 8. Save treatment recommendations from LLM
      if (report?.id && llm.sections.TREATMENT_SUGGESTIONS) {
        const recs = llm.sections.TREATMENT_SUGGESTIONS
          .split(/\n|;/)
          .map(s => s.trim())
          .filter(s => s.length > 10)
          .slice(0, 5);

        for (const rec of recs) {
          await supabase.from('treatment_recommendations').insert({
            report_id: report.id,
            patient_id: patientId,
            recommendation_text: rec,
            recommendation_type: 'follow_up',
            priority: overallRisk === 'High' ? 'urgent' : overallRisk === 'Moderate' ? 'medium' : 'low',
          });
        }
      }

      // 9. Create alert if critical
      if (vlmHasCritical || llmRisk === 'High') {
        await supabase.from('alerts').insert({
          patient_id: patientId,
          analysis_id: analysisId,
          doctor_id: doctorId,
          severity: 'high',
          title: vlmHasCritical ? 'Critical VLM Finding — Immediate Review Required' : 'High Clinical Risk — Review Required',
          description: summary,
          status: 'sent',
        });
      }

      return analysisId;
    } catch (err) {
      console.error('Database save error:', err);
      return null;
    }
  };

  const saveAnomalies = async (vlm: VLMPipelineResult, analysisId: string | null, patientId: string | null) => {
    const anomalies = [];

    if (vlm.brain_anomaly && !vlm.brain_anomaly.is_normal) {
      anomalies.push({
        analysis_id: analysisId,
        patient_id: patientId,
        anomaly_name: vlm.brain_anomaly.label,
        anomaly_region: 'brain',
        severity_level: vlm.brain_anomaly.confidence > 0.8 ? 'high' : 'medium',
        confidence_score: vlm.brain_anomaly.confidence,
        explanation: vlm.brain_anomaly.raw_text,
        is_critical: true,
      });
    }

    if (vlm.nt_markers && vlm.nt_markers.risk === 'High') {
      anomalies.push({
        analysis_id: analysisId,
        patient_id: patientId,
        anomaly_name: `Enlarged NT Thickness (${vlm.nt_markers.nt_thickness_mm.toFixed(2)}mm)`,
        anomaly_region: 'nuchal_translucency',
        severity_level: 'high',
        confidence_score: 0.95,
        explanation: vlm.nt_markers.raw_text,
        is_critical: true,
      });
    }

    if (vlm.heart_segmentation && vlm.heart_segmentation.ctr > 0.55) {
      anomalies.push({
        analysis_id: analysisId,
        patient_id: patientId,
        anomaly_name: `Cardiomegaly (CTR: ${vlm.heart_segmentation.ctr.toFixed(3)})`,
        anomaly_region: 'heart',
        severity_level: 'high',
        confidence_score: 0.9,
        explanation: vlm.heart_segmentation.raw_text,
        is_critical: true,
      });
    }

    if (anomalies.length > 0) {
      await supabase.from('detected_anomalies').insert(anomalies);
    }
  };

  const buildSummary = (vlm: VLMPipelineResult, llm: MedMOReport): string => {
    const parts: string[] = [];
    parts.push(`Plane: ${vlm.plane.label} (${(vlm.plane.confidence * 100).toFixed(0)}% confidence).`);
    if (vlm.brain_anomaly) parts.push(`Brain: ${vlm.brain_anomaly.label}.`);
    if (vlm.nt_markers) parts.push(`NT: ${vlm.nt_markers.nt_thickness_mm}mm (${vlm.nt_markers.risk} risk).`);
    if (vlm.heart_segmentation) parts.push(`CTR: ${vlm.heart_segmentation.ctr.toFixed(3)} — ${vlm.heart_segmentation.interpretation}`);
    parts.push(`Clinical risk: ${llm.risk_level}.`);
    return parts.join(' ');
  };

  const buildFindingsLabel = (vlm: VLMPipelineResult, llm: MedMOReport): string => {
    if (vlm.brain_anomaly && !vlm.brain_anomaly.is_normal) return vlm.brain_anomaly.label;
    if (vlm.nt_markers && vlm.nt_markers.risk === 'High') return 'NT Enlarged';
    if (vlm.heart_segmentation && vlm.heart_segmentation.ctr > 0.55) return 'Cardiomegaly';
    if (llm.risk_level === 'High') return 'High Clinical Risk';
    return 'Normal';
  };

  const inputClassName = "w-full bg-gray-50 border border-gray-200 rounded-md px-3 py-2 text-sm font-mono text-gray-800 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-colors";
  const selectClassName = "w-full bg-gray-50 border border-gray-200 rounded-md px-3 py-2 text-sm font-mono text-gray-800 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-colors appearance-none";
  const labelClassName = "text-xs font-semibold text-gray-600 uppercase tracking-wider mb-1.5 block";

  // ─── Progress steps display ────────────────────────────────────────────────
  const steps = [
    { key: 'uploading', label: 'Uploading image' },
    { key: 'vlm', label: 'VLM plane classification & routing' },
    { key: 'llm', label: 'LLM clinical risk assessment' },
    { key: 'saving', label: 'Saving to database' },
  ];

  const isAnalyzing = ['uploading', 'vlm', 'llm', 'saving'].includes(state.step);

  return (
    <div className="max-w-6xl mx-auto space-y-6 sm:space-y-8">
      <div className="mb-6 sm:mb-8">
        <h1 className="text-xl sm:text-2xl font-semibold text-gray-900 tracking-tight mb-2">Dual-Model Analysis</h1>
        <p className="text-gray-500 text-sm max-w-2xl">
          Upload an ultrasound image for VLM analysis and provide clinical data for LLM risk assessment.
        </p>
      </div>

      {state.step === 'done' && results ? (
        <ResultsView
          results={results}
          clinicalData={clinicalData}
          onReset={() => { setState({ step: 'idle', progress: 0 }); setResults(null); setFile(null); }}
        />
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* Image Upload (VLM) */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="lg:col-span-4 bg-white p-5 sm:p-6 rounded-xl border border-gray-200 shadow-sm"
          >
            <div className="flex items-center gap-2 mb-6">
              <div className="w-8 h-8 rounded-lg bg-blue-50 text-blue-600 flex items-center justify-center">
                <FileText size={18} />
              </div>
              <h3 className="text-lg font-medium text-gray-900">1. VLM Input</h3>
            </div>

            {!file ? (
              <div
                className={`border-2 border-dashed rounded-xl p-6 sm:p-8 text-center transition-colors ${dragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'}`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
              >
                <div className="w-12 h-12 rounded-full bg-gray-50 flex items-center justify-center text-gray-400 mx-auto mb-4">
                  <UploadCloud size={24} />
                </div>
                <p className="text-sm font-medium text-gray-900 mb-1">Drag & drop scan</p>
                <p className="text-xs text-gray-500 mb-6">DICOM, JPEG, PNG (Max 50MB)</p>
                <label className="px-5 py-2.5 bg-white border border-gray-200 text-gray-700 rounded-lg text-sm font-medium hover:bg-gray-50 transition-colors cursor-pointer inline-block shadow-sm">
                  Browse Files
                  <input
                    type="file"
                    className="hidden"
                    accept="image/*,.dcm"
                    onChange={(e) => e.target.files && setFile(e.target.files[0])}
                  />
                </label>
              </div>
            ) : (
              <div className="border border-gray-200 rounded-xl p-4 sm:p-5 relative bg-gray-50/50">
                <button onClick={() => setFile(null)} className="absolute top-3 right-3 text-gray-400 hover:text-red-500 transition-colors">
                  <X size={18} />
                </button>
                <div className="flex items-center gap-3 sm:gap-4">
                  <div className="w-10 h-10 rounded-lg bg-blue-100 text-blue-600 flex items-center justify-center flex-shrink-0">
                    <File size={20} />
                  </div>
                  <div className="min-w-0">
                    <p className="text-sm font-medium text-gray-900 truncate">{file.name}</p>
                    <p className="text-xs text-gray-500">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                  </div>
                </div>
              </div>
            )}

            {/* Progress */}
            {isAnalyzing && (
              <div className="mt-4 space-y-3">
                <div className="w-full bg-gray-100 rounded-full h-1.5">
                  <div
                    className="bg-blue-500 h-1.5 rounded-full transition-all duration-500"
                    style={{ width: `${state.progress}%` }}
                  />
                </div>
                <div className="space-y-1.5">
                  {steps.map((s, i) => {
                    const stepIndex = steps.findIndex(x => x.key === state.step);
                    const isDone = i < stepIndex;
                    const isCurrent = i === stepIndex;
                    return (
                      <div key={s.key} className={`flex items-center gap-2 text-xs ${isDone ? 'text-emerald-600' : isCurrent ? 'text-blue-600 font-medium' : 'text-gray-400'}`}>
                        {isDone ? <CheckCircle2 size={12} /> : isCurrent ? <div className="w-3 h-3 border-2 border-current border-t-transparent rounded-full animate-spin" /> : <div className="w-3 h-3 rounded-full border border-current" />}
                        {s.label}
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {state.step === 'error' && (
              <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg text-xs text-red-600">
                {state.error}
              </div>
            )}

            <div className="mt-6 pt-6 border-t border-gray-100">
              <button
                onClick={handleAnalyze}
                disabled={!file || isAnalyzing}
                className={`w-full px-6 py-3 rounded-lg text-sm font-medium transition-all duration-200 flex items-center justify-center gap-2 shadow-sm ${
                  !file ? 'bg-gray-100 text-gray-400 cursor-not-allowed' :
                  isAnalyzing ? 'bg-gray-800 text-white cursor-wait' :
                  'bg-gray-900 text-white hover:bg-gray-800'
                }`}
              >
                {isAnalyzing ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    Analysing...
                  </>
                ) : (
                  <>
                    <BrainCircuit size={16} />
                    Run VLM + LLM Analysis
                  </>
                )}
              </button>
            </div>
          </motion.div>

          {/* Clinical Data (LLM) */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="lg:col-span-8 bg-white p-5 sm:p-6 rounded-xl border border-gray-200 shadow-sm"
          >
            <div className="flex items-center gap-2 mb-6 pb-4 border-b border-gray-100">
              <div className="w-8 h-8 rounded-lg bg-emerald-50 text-emerald-600 flex items-center justify-center">
                <Activity size={18} />
              </div>
              <div>
                <h3 className="text-lg font-medium text-gray-900">2. LLM Clinical Data</h3>
                <p className="text-xs text-gray-500 mt-0.5">Comprehensive patient profile for risk assessment</p>
              </div>
            </div>

            <div className="space-y-8">
              {/* Vitals & Demographics */}
              <section>
                <div className="flex items-center gap-2 mb-4">
                  <HeartPulse size={16} className="text-gray-400" />
                  <h4 className="text-sm font-semibold text-gray-800">Vitals & Demographics</h4>
                </div>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                  <div>
                    <label className={labelClassName}>Age</label>
                    <input type="number" name="age" value={clinicalData.age} onChange={handleInputChange} className={inputClassName} placeholder="e.g. 28" />
                  </div>
                  <div>
                    <label className={labelClassName}>BMI</label>
                    <input type="number" step="0.1" name="bmi" value={clinicalData.bmi} onChange={handleInputChange} className={inputClassName} placeholder="e.g. 24.5" />
                  </div>
                  <div>
                    <label className={labelClassName}>BP (mmHg)</label>
                    <input type="text" name="blood_pressure" value={clinicalData.blood_pressure} onChange={handleInputChange} className={inputClassName} placeholder="120/80" />
                  </div>
                  <div>
                    <label className={labelClassName}>Gest. Age (Wks)</label>
                    <input type="number" name="gestational_age" value={clinicalData.gestational_age} onChange={handleInputChange} className={inputClassName} placeholder="e.g. 20" />
                  </div>
                </div>
              </section>

              {/* Obstetric History */}
              <section>
                <div className="flex items-center gap-2 mb-4">
                  <Activity size={16} className="text-gray-400" />
                  <h4 className="text-sm font-semibold text-gray-800">Obstetric History</h4>
                </div>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                  <div>
                    <label className={labelClassName}>Prev. C-Section</label>
                    <select name="previous_c_section" value={clinicalData.previous_c_section} onChange={handleInputChange} className={selectClassName}>
                      <option>No</option><option>Yes</option>
                    </select>
                  </div>
                  <div>
                    <label className={labelClassName}>Miscarriages</label>
                    <input type="number" name="previous_miscarriages" value={clinicalData.previous_miscarriages} onChange={handleInputChange} className={inputClassName} min="0" />
                  </div>
                  <div>
                    <label className={labelClassName}>Preterm Birth</label>
                    <select name="previous_preterm_birth" value={clinicalData.previous_preterm_birth} onChange={handleInputChange} className={selectClassName}>
                      <option>No</option><option>Yes</option>
                    </select>
                  </div>
                  <div>
                    <label className={labelClassName}>Multiple Preg.</label>
                    <select name="multiple_pregnancy" value={clinicalData.multiple_pregnancy} onChange={handleInputChange} className={selectClassName}>
                      <option>No</option><option>Yes</option>
                    </select>
                  </div>
                </div>
              </section>

              {/* Medical History & Habits */}
              <section>
                <div className="flex items-center gap-2 mb-4">
                  <FileText size={16} className="text-gray-400" />
                  <h4 className="text-sm font-semibold text-gray-800">Medical History & Habits</h4>
                </div>
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
                  <div>
                    <label className={labelClassName}>Chronic HTN</label>
                    <select name="chronic_hypertension" value={clinicalData.chronic_hypertension} onChange={handleInputChange} className={selectClassName}>
                      <option>No</option><option>Yes</option>
                    </select>
                  </div>
                  <div>
                    <label className={labelClassName}>Diabetes</label>
                    <select name="diabetes" value={clinicalData.diabetes} onChange={handleInputChange} className={selectClassName}>
                      <option>No</option><option>Type 1</option><option>Type 2</option>
                    </select>
                  </div>
                  <div>
                    <label className={labelClassName}>Gest. Diabetes</label>
                    <select name="gestational_diabetes" value={clinicalData.gestational_diabetes} onChange={handleInputChange} className={selectClassName}>
                      <option>No</option><option>Yes</option>
                    </select>
                  </div>
                  <div>
                    <label className={labelClassName}>Preeclampsia Hist.</label>
                    <select name="preeclampsia_history" value={clinicalData.preeclampsia_history} onChange={handleInputChange} className={selectClassName}>
                      <option>No</option><option>Yes</option>
                    </select>
                  </div>
                  <div>
                    <label className={labelClassName}>Smoking</label>
                    <select name="smoking" value={clinicalData.smoking} onChange={handleInputChange} className={selectClassName}>
                      <option>No</option><option>Current</option><option>Former</option>
                    </select>
                  </div>
                  <div>
                    <label className={labelClassName}>Alcohol Use</label>
                    <select name="alcohol_use" value={clinicalData.alcohol_use} onChange={handleInputChange} className={selectClassName}>
                      <option>No</option><option>Occasional</option><option>Regular</option>
                    </select>
                  </div>
                  <div className="col-span-2 sm:col-span-3">
                    <label className={labelClassName}>Family History</label>
                    <input type="text" name="family_history" value={clinicalData.family_history} onChange={handleInputChange} className={inputClassName} placeholder="Relevant family medical history..." />
                  </div>
                </div>
              </section>

              {/* Lab Results */}
              <section>
                <div className="flex items-center gap-2 mb-4">
                  <TestTube size={16} className="text-gray-400" />
                  <h4 className="text-sm font-semibold text-gray-800">Lab Results & Assessment</h4>
                </div>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                  <div>
                    <label className={labelClassName}>Hb Level (g/dL)</label>
                    <input type="number" step="0.1" name="hb_level" value={clinicalData.hb_level} onChange={handleInputChange} className={inputClassName} placeholder="e.g. 12.5" />
                  </div>
                  <div>
                    <label className={labelClassName}>Urine Protein</label>
                    <select name="urine_protein" value={clinicalData.urine_protein} onChange={handleInputChange} className={selectClassName}>
                      <option value="">Select...</option>
                      <option>Negative</option><option>Trace</option><option>1+</option><option>2+</option><option>3+</option><option>4+</option>
                    </select>
                  </div>
                  <div>
                    <label className={labelClassName}>Blood Glucose</label>
                    <input type="number" name="blood_glucose" value={clinicalData.blood_glucose} onChange={handleInputChange} className={inputClassName} placeholder="mg/dL" />
                  </div>
                  <div>
                    <label className={labelClassName}>Assessed Risk</label>
                    <select name="risk_level" value={clinicalData.risk_level} onChange={handleInputChange} className={`${selectClassName} font-semibold ${
                      clinicalData.risk_level === 'High' ? 'text-red-600 bg-red-50 border-red-200' :
                      clinicalData.risk_level === 'Moderate' ? 'text-amber-600 bg-amber-50 border-amber-200' :
                      'text-emerald-600 bg-emerald-50 border-emerald-200'
                    }`}>
                      <option value="Low">Low Risk</option>
                      <option value="Moderate">Moderate Risk</option>
                      <option value="High">High Risk</option>
                    </select>
                  </div>
                </div>
              </section>
            </div>
          </motion.div>
        </div>
      )}
    </div>
  );
}

// ─── Results View Component ──────────────────────────────────────────────────
function ResultsView({ results, clinicalData, onReset }: {
  results: Results;
  clinicalData: ClinicalFormData;
  onReset: () => void;
}) {
  const { vlm, llm } = results;
  const riskColor = llm.risk_level === 'High' ? 'red' : llm.risk_level === 'Moderate' ? 'amber' : 'emerald';

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="bg-white p-5 sm:p-8 rounded-xl border border-gray-200 shadow-sm"
    >
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center gap-4 mb-6 sm:mb-8 pb-6 sm:pb-8 border-b border-gray-100">
        <div className={`w-12 h-12 rounded-full flex items-center justify-center flex-shrink-0 ${
          llm.risk_level === 'High' ? 'bg-red-100 text-red-600' :
          llm.risk_level === 'Moderate' ? 'bg-amber-100 text-amber-600' : 'bg-emerald-100 text-emerald-600'
        }`}>
          <AlertCircle size={24} />
        </div>
        <div>
          <h2 className="text-lg sm:text-xl font-semibold text-gray-900 tracking-tight">Analysis Complete</h2>
          <p className="text-sm text-gray-500 mt-1">
            VLM: {vlm.plane.label} plane detected • Route: {vlm.routed_to.toUpperCase()} • LLM Risk: {llm.risk_level}
            {results.reportCode && ` • ${results.reportCode}`}
          </p>
        </div>
        <div className="sm:ml-auto">
          <span className={`px-3 py-1.5 rounded-full text-sm font-semibold border ${
            riskColor === 'red' ? 'bg-red-50 text-red-700 border-red-200' :
            riskColor === 'amber' ? 'bg-amber-50 text-amber-700 border-amber-200' :
            'bg-emerald-50 text-emerald-700 border-emerald-200'
          }`}>
            {llm.risk_level} Risk
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 sm:gap-10">
        {/* VLM Results */}
        <div>
          <h3 className="text-sm font-semibold text-gray-900 mb-4">VLM Image Analysis</h3>
          <div className="space-y-3">
            {/* Plane */}
            <div className="p-3.5 rounded-lg border border-blue-200 bg-blue-50 flex items-start gap-3">
              <CheckCircle2 className="text-blue-600 flex-shrink-0 mt-0.5" size={18} />
              <div className="flex items-center">
                <p className="text-sm font-medium text-blue-900">Plane: {vlm.plane.label}</p>
              </div>
            </div>

            {/* Brain */}
            {vlm.brain_anomaly && (
              <div className={`p-4 rounded-lg border ${vlm.brain_anomaly.is_normal ? 'border-emerald-200 bg-emerald-50' : 'border-red-200 bg-red-50'}`}>
                <div className="flex justify-between items-start gap-2 mb-1">
                  <h4 className={`font-medium text-sm ${vlm.brain_anomaly.is_normal ? 'text-emerald-900' : 'text-red-900'}`}>
                    Brain: {vlm.brain_anomaly.label}
                  </h4>
                  <span className={`px-2 py-0.5 rounded text-xs font-medium ${vlm.brain_anomaly.is_normal ? 'bg-emerald-200/50 text-emerald-800' : 'bg-red-200/50 text-red-800'}`}>
                    {(vlm.brain_anomaly.confidence * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            )}

            {/* NT */}
            {vlm.nt_markers && (
              <div className={`p-4 rounded-lg border ${vlm.nt_markers.risk === 'High' ? 'border-red-200 bg-red-50' : 'border-emerald-200 bg-emerald-50'}`}>
                <div className="flex justify-between items-start gap-2 mb-1">
                  <h4 className={`font-medium text-sm ${vlm.nt_markers.risk === 'High' ? 'text-red-900' : 'text-emerald-900'}`}>
                    NT Thickness: {vlm.nt_markers.nt_thickness_mm}mm
                  </h4>
                  <span className={`px-2 py-0.5 rounded text-xs font-medium ${vlm.nt_markers.risk === 'High' ? 'bg-red-200/50 text-red-800' : 'bg-emerald-200/50 text-emerald-800'}`}>
                    {vlm.nt_markers.risk} Risk
                  </span>
                </div>
                <p className={`text-xs ${vlm.nt_markers.risk === 'High' ? 'text-red-700' : 'text-emerald-700'} mb-2`}>
                  Nasal bone: {vlm.nt_markers.nasal_bone_present ? 'Present' : 'Absent'}
                  {vlm.nt_markers.risk === 'High' && ' — NT ≥ 3.5mm. Down syndrome screening recommended.'}
                </p>
                {vlm.nt_markers.heatmap_url && (
                  <div className="mt-2 rounded overflow-hidden border border-gray-200 max-h-48 flex justify-center bg-black/5">
                    <img src={vlm.nt_markers.heatmap_url} alt="NT Heatmap" className="max-h-48 object-contain" />
                  </div>
                )}
              </div>
            )}

            {/* Heart */}
            {vlm.heart_segmentation && (
              <div className={`p-4 rounded-lg border ${vlm.heart_segmentation.ctr > 0.55 ? 'border-amber-200 bg-amber-50' : 'border-emerald-200 bg-emerald-50'}`}>
                <div className="flex justify-between items-start gap-2 mb-1">
                  <h4 className={`font-medium text-sm ${vlm.heart_segmentation.ctr > 0.55 ? 'text-amber-900' : 'text-emerald-900'}`}>
                    CTR: {vlm.heart_segmentation.ctr.toFixed(3)}
                  </h4>
                  <span className={`px-2 py-0.5 rounded text-xs font-medium ${vlm.heart_segmentation.ctr > 0.55 ? 'bg-amber-200/50 text-amber-800' : 'bg-emerald-200/50 text-emerald-800'}`}>
                    {vlm.heart_segmentation.ctr > 0.55 ? 'Abnormal' : 'Normal'}
                  </span>
                </div>
                <p className={`text-xs ${vlm.heart_segmentation.ctr > 0.55 ? 'text-amber-800' : 'text-emerald-800'} mb-2`}>
                  {vlm.heart_segmentation.interpretation}
                </p>
                {vlm.heart_segmentation.overlay_url && (
                  <div className="mt-2 rounded overflow-hidden border border-gray-200 max-h-48 flex justify-center bg-black/5">
                    <img src={vlm.heart_segmentation.overlay_url} alt="Heart Segmentation Overlay" className="max-h-48 object-contain" />
                  </div>
                )}
              </div>
            )}

            {/* Other plane */}
            {vlm.routed_to === 'other' && (
              <div className="p-3.5 rounded-lg border border-gray-200 bg-gray-50">
                <p className="text-sm text-gray-600">No anomaly detection available for this plane type. Plane classified as: <strong>{vlm.plane.label}</strong></p>
              </div>
            )}
          </div>

          {/* LLM summary */}
          <h3 className="text-sm font-semibold text-gray-900 mt-8 mb-4">LLM Clinical Assessment</h3>
          <div className={`p-4 rounded-lg border ${
            llm.risk_level === 'High' ? 'border-red-200 bg-red-50' :
            llm.risk_level === 'Moderate' ? 'border-amber-200 bg-amber-50' :
            'border-blue-200 bg-blue-50'
          }`}>
            <div className="flex items-center gap-2 mb-2">
              <BrainCircuit className={llm.risk_level === 'High' ? 'text-red-600' : llm.risk_level === 'Moderate' ? 'text-amber-600' : 'text-blue-600'} size={18} />
              <h4 className={`font-medium text-sm ${llm.risk_level === 'High' ? 'text-red-900' : llm.risk_level === 'Moderate' ? 'text-amber-900' : 'text-blue-900'}`}>
                {llm.risk_level} Risk Profile
              </h4>
            </div>
            {llm.risk_factors.length > 0 && (
              <ul className="text-xs space-y-1 mt-2">
                {llm.risk_factors.slice(0, 4).map((rf, i) => (
                  <li key={i} className={`${llm.risk_level === 'High' ? 'text-red-800' : llm.risk_level === 'Moderate' ? 'text-amber-800' : 'text-blue-800'}`}>• {rf}</li>
                ))}
              </ul>
            )}
          </div>
        </div>

        {/* Recommendations & Actions */}
        <div>
          <h3 className="text-sm font-semibold text-gray-900 mb-4">Treatment Recommendations</h3>
          <div className="prose prose-sm text-gray-600 bg-gray-50 rounded-lg p-4 border border-gray-100">
            <p className="text-xs text-gray-700 leading-relaxed whitespace-pre-line">
              {llm.sections.TREATMENT_SUGGESTIONS || 'Continue routine prenatal care.'}
            </p>
          </div>

          {llm.sections.MONITORING && (
            <>
              <h3 className="text-sm font-semibold text-gray-900 mt-6 mb-3">Monitoring Plan</h3>
              <div className="bg-blue-50 border border-blue-100 rounded-lg p-4">
                <p className="text-xs text-blue-800 leading-relaxed whitespace-pre-line">{llm.sections.MONITORING}</p>
              </div>
            </>
          )}

          {llm.sections.REFERRAL && (
            <>
              <h3 className="text-sm font-semibold text-gray-900 mt-6 mb-3">Referral</h3>
              <div className="bg-gray-50 border border-gray-100 rounded-lg p-4">
                <p className="text-xs text-gray-700 leading-relaxed">{llm.sections.REFERRAL}</p>
              </div>
            </>
          )}

          <div className="mt-8 sm:mt-10 flex flex-col sm:flex-row gap-3">
            <button
              onClick={onReset}
              className="w-full sm:w-auto px-4 py-2.5 sm:py-2 bg-white border border-gray-300 text-gray-700 rounded-lg text-sm font-medium hover:bg-gray-50 transition-colors shadow-sm flex items-center gap-2 justify-center"
            >
              <RotateCcw size={15} />
              New Scan
            </button>
            {results.reportCode && (
              <div className="text-xs text-gray-400 flex items-center gap-1 pt-2">
                <CheckCircle2 size={12} className="text-emerald-500" />
                Report saved: {results.reportCode}
              </div>
            )}
          </div>
        </div>
      </div>
    </motion.div>
  );
}
