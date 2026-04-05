// LLM Integration — MedMO-8B Clinical Risk Assessment
// Based on medmo-llm notebook logic, ported for browser/API use
// The LLM (MBZUAI/MedMO-8B) should be hosted on HuggingFace Inference API or Spaces
// If VITE_LLM_API_URL is empty, the prompt is returned for manual use

const LLM_BASE = import.meta.env.VITE_LLM_API_URL || '';

export interface ClinicalFormData {
  age: string;
  bmi: string;
  blood_pressure: string;
  gestational_age: string;
  previous_c_section: string;
  previous_miscarriages: string;
  previous_preterm_birth: string;
  multiple_pregnancy: string;
  chronic_hypertension: string;
  diabetes: string;
  gestational_diabetes: string;
  preeclampsia_history: string;
  smoking: string;
  alcohol_use: string;
  family_history: string;
  hb_level: string;
  urine_protein: string;
  blood_glucose: string;
  risk_level: string;
}

export interface MedMOReport {
  risk_level: 'Low' | 'Moderate' | 'High' | 'Unknown';
  risk_factors: string[];
  sections: {
    RISK_FACTORS: string;
    RISK_LEVEL: string;
    MATERNAL_RISKS: string;
    FETAL_RISKS: string;
    MONITORING: string;
    TREATMENT_SUGGESTIONS: string;
    COUNSELLING: string;
    REFERRAL: string;
  };
  raw_response: string;
}

// ─── Build the prompt text from clinical form data ─────────────────────────────
// Matches medmo-llm notebook: build_prompt() → key: value structure

export function buildPatientText(data: ClinicalFormData): string {
  const lines = [
    `patient_age_years: ${data.age || 'N/A'}`,
    `bmi: ${data.bmi || 'N/A'}`,
    `blood_pressure_mmhg: ${data.blood_pressure || 'N/A'}`,
    `gestational_age_weeks: ${data.gestational_age || 'N/A'}`,
    `previous_c_section: ${data.previous_c_section}`,
    `previous_miscarriages: ${data.previous_miscarriages}`,
    `previous_preterm_birth: ${data.previous_preterm_birth}`,
    `multiple_pregnancy: ${data.multiple_pregnancy}`,
    `chronic_hypertension: ${data.chronic_hypertension}`,
    `diabetes: ${data.diabetes}`,
    `gestational_diabetes: ${data.gestational_diabetes}`,
    `preeclampsia_history: ${data.preeclampsia_history}`,
    `smoking: ${data.smoking}`,
    `alcohol_use: ${data.alcohol_use}`,
    `family_history: ${data.family_history || 'None reported'}`,
    `hb_level_g_dl: ${data.hb_level || 'N/A'}`,
    `urine_protein: ${data.urine_protein || 'N/A'}`,
    `blood_glucose_mg_dl: ${data.blood_glucose || 'N/A'}`,
    `doctor_assessed_risk: ${data.risk_level}`,
  ];
  return lines.join('\n');
}

export function buildPrompt(patientText: string): string {
  return `I am a doctor reviewing a pregnant patient. Here is her clinical data:
${patientText}

Please provide a complete clinical assessment covering all of the following 8 points:
1. What are the risk factors?
2. What is the overall risk level (Low, Moderate, or High)?
3. What are the maternal complications to watch for?
4. What are the fetal risks?
5. What monitoring and investigations are needed?
6. What treatment is recommended?
7. What should I counsel the patient about?
8. Is specialist referral needed?

Format each section clearly labeled: RISK FACTORS:, RISK LEVEL:, MATERNAL RISKS:, FETAL RISKS:, MONITORING:, TREATMENT SUGGESTIONS:, COUNSELLING:, REFERRAL:`;
}

// ─── Parse the model response into 8 sections ────────────────────────────────
// Matches medmo-llm notebook: parse_sections()

export function parseSections(rawResponse: string): MedMOReport['sections'] {
  const sectionNames = [
    'RISK FACTORS',
    'RISK LEVEL',
    'MATERNAL RISKS',
    'FETAL RISKS',
    'MONITORING',
    'TREATMENT SUGGESTIONS',
    'COUNSELLING',
    'REFERRAL',
  ] as const;

  const cleanName = (s: string) => s.replace(/\s/g, '_');

  const result: Record<string, string> = {};

  for (let i = 0; i < sectionNames.length; i++) {
    const current = sectionNames[i];
    const next = sectionNames[i + 1];
    const pattern = next
      ? new RegExp(`${current}:([\\s\\S]*?)(?=${next}:)`, 'i')
      : new RegExp(`${current}:([\\s\\S]*)$`, 'i');
    const match = rawResponse.match(pattern);
    result[cleanName(current)] = match ? match[1].trim() : '';
  }

  return result as MedMOReport['sections'];
}

// ─── Extract risk level string from raw response ──────────────────────────────

export function extractRiskLevel(rawResponse: string): MedMOReport['risk_level'] {
  const patterns = [
    /RISK LEVEL:\s*(Low|Moderate|High)/i,
    /overall risk[:\s]+(Low|Moderate|High)/i,
    /risk is\s+(Low|Moderate|High)/i,
    /\b(High|Moderate|Low)\s+Risk\b/i,
  ];
  for (const p of patterns) {
    const m = rawResponse.match(p);
    if (m) {
      const val = m[1].charAt(0).toUpperCase() + m[1].slice(1).toLowerCase();
      if (val === 'Low' || val === 'Moderate' || val === 'High') return val;
    }
  }
  return 'Unknown';
}

// ─── Extract risk factors as array ───────────────────────────────────────────

export function extractRiskFactors(rawResponse: string): string[] {
  const sections = parseSections(rawResponse);
  const rf = sections.RISK_FACTORS;
  if (!rf) return [];

  return rf
    .split(/\n|;|\d+\.|•|-/)
    .map((s) => s.trim())
    .filter((s) => s.length > 10)
    .slice(0, 8);
}

// ─── Main inference call — POSTs to LLM API endpoint ─────────────────────────
// Expected API: POST {LLM_BASE}/generate with body { inputs: string }
// Response: [{ generated_text: string }]

export async function runMedMOAnalysis(data: ClinicalFormData): Promise<MedMOReport> {
  const patientText = buildPatientText(data);
  const prompt = buildPrompt(patientText);

  // If no LLM endpoint is set, return a structured placeholder so UI can still work
  if (!LLM_BASE) {
    return generateFallbackReport(data, patientText);
  }

  const res = await fetch(`${LLM_BASE}/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      inputs: prompt,
      parameters: {
        max_new_tokens: 1500,
        do_sample: true,
        temperature: 0.1,
        top_p: 0.9,
        repetition_penalty: 1.05,
      },
    }),
  });

  if (!res.ok) {
    throw new Error(`LLM API error: ${res.status} ${res.statusText}`);
  }

  const json = await res.json();
  // HuggingFace Inference API format: [{ generated_text: "..." }]
  const rawResponse: string = Array.isArray(json)
    ? json[0]?.generated_text || ''
    : json?.generated_text || json?.text || '';

  const sections = parseSections(rawResponse);
  const risk_level = extractRiskLevel(rawResponse);
  const risk_factors = extractRiskFactors(rawResponse);

  return { risk_level, risk_factors, sections, raw_response: rawResponse };
}

// ─── Fallback report when LLM API is unavailable ─────────────────────────────
// Uses rule-based logic from the clinical form data

function generateFallbackReport(data: ClinicalFormData, patientText: string): MedMOReport {
  const riskFactors: string[] = [];

  if (parseInt(data.age) > 35) riskFactors.push('Advanced maternal age (>35 years)');
  if (parseFloat(data.bmi) > 30) riskFactors.push('Obesity (BMI >30)');
  if (data.chronic_hypertension === 'Yes') riskFactors.push('Chronic hypertension');
  if (data.diabetes !== 'No') riskFactors.push(`Pre-existing diabetes (${data.diabetes})`);
  if (data.gestational_diabetes === 'Yes') riskFactors.push('Gestational diabetes');
  if (data.preeclampsia_history === 'Yes') riskFactors.push('Previous preeclampsia');
  if (data.previous_preterm_birth === 'Yes') riskFactors.push('Previous preterm birth');
  if (data.smoking !== 'No') riskFactors.push(`Smoking (${data.smoking})`);
  if (data.alcohol_use !== 'No') riskFactors.push(`Alcohol use (${data.alcohol_use})`);
  if (data.multiple_pregnancy === 'Yes') riskFactors.push('Multiple pregnancy');
  if (parseFloat(data.hb_level) < 10) riskFactors.push('Anaemia (Hb <10 g/dL)');
  if (data.urine_protein && !['Negative', ''].includes(data.urine_protein)) riskFactors.push(`Proteinuria (${data.urine_protein})`);

  const risk_level: MedMOReport['risk_level'] =
    riskFactors.length >= 4 ? 'High' : riskFactors.length >= 2 ? 'Moderate' : 'Low';

  const sections: MedMOReport['sections'] = {
    RISK_FACTORS: riskFactors.join('\n') || 'No significant risk factors identified.',
    RISK_LEVEL: risk_level,
    MATERNAL_RISKS:
      risk_level === 'High'
        ? 'Risk of preeclampsia, gestational hypertension, postpartum haemorrhage. Close monitoring required.'
        : 'Standard maternal risks. Routine prenatal care recommended.',
    FETAL_RISKS:
      risk_level === 'High'
        ? 'Intrauterine growth restriction (IUGR), preterm delivery, fetal distress. Serial growth scans advised.'
        : 'No elevated fetal risks identified based on clinical data alone.',
    MONITORING:
      'Regular BP monitoring, serial growth scans every 4 weeks from 28 weeks, urine dipstick at each visit, fasting glucose at 24-28 weeks.',
    TREATMENT_SUGGESTIONS:
      risk_level === 'High'
        ? 'Consider low-dose aspirin (75-150 mg) if indicated. Optimise glycaemic control. Iron supplementation if anaemic.'
        : 'Continue routine prenatal care. Iron and folate supplementation. Healthy eating and moderate exercise.',
    COUNSELLING:
      'Advise on warning signs: headache, visual disturbance, epigastric pain, reduced fetal movements. Stop smoking if applicable. Attend all antenatal appointments.',
    REFERRAL:
      risk_level === 'High'
        ? 'Refer to maternal-fetal medicine specialist. Cardiology referral if cardiac findings on ultrasound.'
        : 'No specialist referral required at this stage. Reassess if clinical condition changes.',
  };

  const raw_response = Object.entries(sections)
    .map(([k, v]) => `${k.replace(/_/g, ' ')}:\n${v}`)
    .join('\n\n');

  return { risk_level, risk_factors: riskFactors, sections, raw_response };
}
