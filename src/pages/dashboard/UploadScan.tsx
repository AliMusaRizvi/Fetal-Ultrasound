import React, { useState } from 'react';
import { motion } from 'motion/react';
import { UploadCloud, File, X, CheckCircle2, AlertCircle, BrainCircuit, Activity, FileText, HeartPulse, TestTube } from 'lucide-react';

export default function UploadScan() {
  const [dragActive, setDragActive] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisComplete, setAnalysisComplete] = useState(false);

  const [clinicalData, setClinicalData] = useState({
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
    risk_level: 'Low'
  });

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target;
    setClinicalData(prev => ({ ...prev, [name]: value }));
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
    }
  };

  const handleAnalyze = () => {
    if (!file) return;
    setIsAnalyzing(true);
    // Simulate analysis
    setTimeout(() => {
      setIsAnalyzing(false);
      setAnalysisComplete(true);
    }, 3000);
  };

  const inputClassName = "w-full bg-gray-50 border border-gray-200 rounded-md px-3 py-2 text-sm font-mono text-gray-800 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-colors";
  const selectClassName = "w-full bg-gray-50 border border-gray-200 rounded-md px-3 py-2 text-sm font-mono text-gray-800 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-colors appearance-none";
  const labelClassName = "text-xs font-semibold text-gray-600 uppercase tracking-wider mb-1.5 block";

  return (
    <div className="max-w-6xl mx-auto space-y-6 sm:space-y-8">
      <div className="mb-6 sm:mb-8">
        <h1 className="text-xl sm:text-2xl font-semibold text-gray-900 tracking-tight mb-2">Dual-Model Analysis</h1>
        <p className="text-gray-500 text-sm max-w-2xl">
          Upload an ultrasound image for VLM (Visual Language Model) analysis and provide clinical data for LLM (Large Language Model) risk assessment.
        </p>
      </div>

      {!analysisComplete ? (
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
                className={`border-2 border-dashed rounded-xl p-6 sm:p-8 text-center transition-colors ${
                  dragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'
                }`}
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
                <button 
                  onClick={() => setFile(null)}
                  className="absolute top-3 right-3 text-gray-400 hover:text-red-500 transition-colors"
                >
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
                    Running Dual Analysis...
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
                    <div className="relative">
                      <select name="previous_c_section" value={clinicalData.previous_c_section} onChange={handleInputChange} className={selectClassName}>
                        <option>No</option>
                        <option>Yes</option>
                      </select>
                    </div>
                  </div>
                  <div>
                    <label className={labelClassName}>Miscarriages</label>
                    <input type="number" name="previous_miscarriages" value={clinicalData.previous_miscarriages} onChange={handleInputChange} className={inputClassName} min="0" />
                  </div>
                  <div>
                    <label className={labelClassName}>Preterm Birth</label>
                    <select name="previous_preterm_birth" value={clinicalData.previous_preterm_birth} onChange={handleInputChange} className={selectClassName}>
                      <option>No</option>
                      <option>Yes</option>
                    </select>
                  </div>
                  <div>
                    <label className={labelClassName}>Multiple Preg.</label>
                    <select name="multiple_pregnancy" value={clinicalData.multiple_pregnancy} onChange={handleInputChange} className={selectClassName}>
                      <option>No</option>
                      <option>Yes</option>
                    </select>
                  </div>
                </div>
              </section>

              {/* Medical History */}
              <section>
                <div className="flex items-center gap-2 mb-4">
                  <FileText size={16} className="text-gray-400" />
                  <h4 className="text-sm font-semibold text-gray-800">Medical History & Habits</h4>
                </div>
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
                  <div>
                    <label className={labelClassName}>Chronic HTN</label>
                    <select name="chronic_hypertension" value={clinicalData.chronic_hypertension} onChange={handleInputChange} className={selectClassName}>
                      <option>No</option>
                      <option>Yes</option>
                    </select>
                  </div>
                  <div>
                    <label className={labelClassName}>Diabetes</label>
                    <select name="diabetes" value={clinicalData.diabetes} onChange={handleInputChange} className={selectClassName}>
                      <option>No</option>
                      <option>Type 1</option>
                      <option>Type 2</option>
                    </select>
                  </div>
                  <div>
                    <label className={labelClassName}>Gest. Diabetes</label>
                    <select name="gestational_diabetes" value={clinicalData.gestational_diabetes} onChange={handleInputChange} className={selectClassName}>
                      <option>No</option>
                      <option>Yes</option>
                    </select>
                  </div>
                  <div>
                    <label className={labelClassName}>Preeclampsia Hist.</label>
                    <select name="preeclampsia_history" value={clinicalData.preeclampsia_history} onChange={handleInputChange} className={selectClassName}>
                      <option>No</option>
                      <option>Yes</option>
                    </select>
                  </div>
                  <div>
                    <label className={labelClassName}>Smoking</label>
                    <select name="smoking" value={clinicalData.smoking} onChange={handleInputChange} className={selectClassName}>
                      <option>No</option>
                      <option>Current</option>
                      <option>Former</option>
                    </select>
                  </div>
                  <div>
                    <label className={labelClassName}>Alcohol Use</label>
                    <select name="alcohol_use" value={clinicalData.alcohol_use} onChange={handleInputChange} className={selectClassName}>
                      <option>No</option>
                      <option>Occasional</option>
                      <option>Regular</option>
                    </select>
                  </div>
                  <div className="col-span-2 sm:col-span-3">
                    <label className={labelClassName}>Family History</label>
                    <input type="text" name="family_history" value={clinicalData.family_history} onChange={handleInputChange} className={inputClassName} placeholder="Relevant family medical history..." />
                  </div>
                </div>
              </section>

              {/* Lab Results & Assessment */}
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
                      <option>Negative</option>
                      <option>Trace</option>
                      <option>1+</option>
                      <option>2+</option>
                      <option>3+</option>
                      <option>4+</option>
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
      ) : (
        /* Results View */
        <motion.div 
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="bg-white p-5 sm:p-8 rounded-xl border border-gray-200 shadow-sm"
        >
          <div className="flex flex-col sm:flex-row sm:items-center gap-4 mb-6 sm:mb-8 pb-6 sm:pb-8 border-b border-gray-100">
            <div className="w-12 h-12 rounded-full bg-amber-100 text-amber-600 flex items-center justify-center flex-shrink-0">
              <AlertCircle size={24} />
            </div>
            <div>
              <h2 className="text-lg sm:text-xl font-semibold text-gray-900 tracking-tight">Dual Analysis Complete: Review Required</h2>
              <p className="text-sm text-gray-500 mt-1">VLM (Image) + LLM (Clinical Data) • Scanned just now</p>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 sm:gap-10">
            <div>
              <h3 className="text-sm font-semibold text-gray-900 mb-4">VLM Image Analysis</h3>
              <div className="space-y-3">
                <div className="p-4 rounded-lg border border-amber-200 bg-amber-50">
                  <div className="flex flex-col sm:flex-row sm:justify-between sm:items-start gap-2 mb-2">
                    <h4 className="font-medium text-amber-900 text-sm">Ventricular Septal Defect (VSD)</h4>
                    <span className="px-2 py-0.5 rounded text-amber-800 bg-amber-200/50 text-xs font-medium self-start sm:self-auto">87% Confidence</span>
                  </div>
                  <p className="text-sm text-amber-800/80">
                    Irregularity detected in the interventricular septum. Further echocardiography recommended.
                  </p>
                </div>
                <div className="p-3.5 rounded-lg border border-emerald-200 bg-emerald-50 flex items-start sm:items-center gap-3">
                  <CheckCircle2 className="text-emerald-600 flex-shrink-0 mt-0.5 sm:mt-0" size={18} />
                  <span className="text-sm font-medium text-emerald-900">Nuchal Translucency (NT) within normal range (1.8mm)</span>
                </div>
              </div>

              <h3 className="text-sm font-semibold text-gray-900 mt-8 mb-4">LLM Clinical Risk Assessment</h3>
              <div className="p-4 rounded-lg border border-blue-200 bg-blue-50">
                <div className="flex items-center gap-2 mb-2">
                  <BrainCircuit className="text-blue-600" size={18} />
                  <h4 className="font-medium text-blue-900 text-sm">Elevated Risk Profile</h4>
                </div>
                <p className="text-sm text-blue-800/80">
                  Based on the clinical data provided (Age: {clinicalData.age || 'N/A'}, BMI: {clinicalData.bmi || 'N/A'}, Gestational Age: {clinicalData.gestational_age || 'N/A'} wks), the patient has an elevated risk profile. The combination of clinical factors and VLM findings suggests close monitoring is required.
                </p>
              </div>
            </div>

            <div>
              <h3 className="text-sm font-semibold text-gray-900 mb-4">Combined Recommendations</h3>
              <div className="prose prose-sm text-gray-600">
                <ul className="space-y-2 list-disc pl-4 marker:text-gray-400">
                  <li>Schedule a detailed fetal echocardiogram within the next 2 weeks based on VLM findings.</li>
                  <li>Refer to pediatric cardiology for consultation.</li>
                  <li>Monitor blood pressure and glucose levels closely due to clinical risk factors.</li>
                  <li>Discuss findings with the parents and provide genetic counseling options.</li>
                  <li>Continue routine prenatal care with closer monitoring of fetal growth.</li>
                </ul>
              </div>

              <div className="mt-8 sm:mt-10 flex flex-col sm:flex-row gap-3">
                <button className="w-full sm:flex-1 px-4 py-2.5 sm:py-2 bg-gray-900 text-white rounded-lg text-sm font-medium hover:bg-gray-800 transition-colors shadow-sm">
                  Download Full Report
                </button>
                <button 
                  onClick={() => setAnalysisComplete(false)}
                  className="w-full sm:w-auto px-4 py-2.5 sm:py-2 bg-white border border-gray-300 text-gray-700 rounded-lg text-sm font-medium hover:bg-gray-50 transition-colors shadow-sm"
                >
                  New Scan
                </button>
              </div>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
}
