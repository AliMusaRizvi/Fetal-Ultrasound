import { useEffect, useState } from 'react';
import { motion } from 'motion/react';
import { Search, Download, Eye } from 'lucide-react';
import { supabase } from '../../lib/supabase';

interface ReportRow {
  id: string;
  report_code: string;
  patient_code: string;
  patient_name?: string;
  findings_label?: string;
  overall_risk?: string;
  is_finalized: boolean;
  pdf_storage_path?: string;
  created_at: string;
}

export default function Reports() {
  const [reports, setReports] = useState<ReportRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [filterStatus, setFilterStatus] = useState('all');

  useEffect(() => {
    loadReports();
  }, []);

  const loadReports = async () => {
    setLoading(true);
    try {
      const { data } = await supabase
        .from('v_reports_list')
        .select('id, report_code, patient_code, patient_name, findings_label, overall_risk, is_finalized, pdf_storage_path, created_at')
        .order('created_at', { ascending: false });

      setReports(data || []);
    } catch (err) {
      console.error('Reports load error:', err);
    } finally {
      setLoading(false);
    }
  };

  const filtered = reports.filter(r => {
    const matchSearch = (r.report_code || '').toLowerCase().includes(search.toLowerCase()) ||
      (r.patient_code || '').toLowerCase().includes(search.toLowerCase());
    const matchStatus = filterStatus === 'all' ||
      (filterStatus === 'normal' && r.findings_label === 'Normal') ||
      (filterStatus === 'review' && r.findings_label !== 'Normal');
    return matchSearch && matchStatus;
  });

  const findingsColor = (label?: string) => {
    if (!label || label === 'Normal') return 'text-emerald-700 bg-emerald-50 border-emerald-200';
    if (label.toLowerCase().includes('high') || label.toLowerCase().includes('enlarged') || label.toLowerCase().includes('cardiomegaly')) return 'text-red-700 bg-red-50 border-red-200';
    return 'text-amber-700 bg-amber-50 border-amber-200';
  };

  const handleReportAction = async (report: ReportRow, action: 'view' | 'download') => {
    if (!report.pdf_storage_path) {
      alert('PDF not yet generated for this report. The analysis might be pending or failed.');
      return;
    }
    
    try {
      const { data, error } = await supabase.storage
        .from('reports')
        .createSignedUrl(report.pdf_storage_path, 60, {
          download: action === 'download'
        });
        
      if (error) throw error;
      
      if (data?.signedUrl) {
        window.open(data.signedUrl, action === 'view' ? '_blank' : '_self');
        
        // Log download activity
        if (action === 'download') {
          await supabase.from('download_logs').insert({
            report_id: report.id,
            doctor_id: localStorage.getItem('userId'),
          });
        }
      }
    } catch (err) {
      console.error(`Failed to ${action} report:`, err);
      alert('Failed to access the report file. It may have been deleted or permissions are missing.');
    }
  };

  return (
    <div className="max-w-6xl mx-auto space-y-6 sm:space-y-8">
      <div className="mb-6 sm:mb-8">
        <h1 className="text-xl sm:text-2xl font-semibold text-gray-900 tracking-tight mb-2">Diagnostic Reports</h1>
        <p className="text-gray-500 text-sm">
          View and download all patient diagnostic reports and AI analysis results.
        </p>
      </div>

      <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden flex flex-col">
        <div className="p-4 sm:p-5 border-b border-gray-100 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 bg-gray-50/50">
          <div className="flex flex-col sm:flex-row gap-3 sm:gap-4 w-full sm:w-auto">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" size={16} />
              <input
                type="text"
                placeholder="Search by report code or patient..."
                value={search}
                onChange={e => setSearch(e.target.value)}
                className="bg-white border border-gray-300 rounded-lg pl-9 pr-3.5 py-2 text-sm focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 shadow-sm w-full sm:w-64"
              />
            </div>
            <select
              value={filterStatus}
              onChange={e => setFilterStatus(e.target.value)}
              className="bg-white border border-gray-300 rounded-lg px-3.5 py-2 text-sm focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 shadow-sm text-gray-700 w-full sm:w-auto"
            >
              <option value="all">All Findings</option>
              <option value="normal">Normal</option>
              <option value="review">Review Required</option>
            </select>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse min-w-[600px]">
            <thead>
              <tr className="bg-white text-xs uppercase tracking-wider text-gray-500 border-b border-gray-100">
                <th className="px-4 sm:px-6 py-4 font-medium">Report</th>
                <th className="px-4 sm:px-6 py-4 font-medium">Patient</th>
                <th className="px-4 sm:px-6 py-4 font-medium">Date</th>
                <th className="px-4 sm:px-6 py-4 font-medium">Findings</th>
                <th className="px-4 sm:px-6 py-4 font-medium text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="text-sm divide-y divide-gray-50">
              {loading ? (
                <tr>
                  <td colSpan={5} className="px-6 py-12 text-center text-gray-400">Loading reports...</td>
                </tr>
              ) : filtered.length === 0 ? (
                <tr>
                  <td colSpan={5} className="px-6 py-12 text-center text-gray-500">
                    {reports.length === 0 ? 'No reports yet. Complete an analysis to generate a report.' : 'No reports match your search.'}
                  </td>
                </tr>
              ) : (
                filtered.map((row, idx) => (
                  <motion.tr
                    initial={{ opacity: 0, y: 5 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: idx * 0.03 }}
                    key={row.id}
                    className="hover:bg-gray-50/50 transition-colors"
                  >
                    <td className="px-4 sm:px-6 py-4 font-mono text-gray-600 text-xs">{row.report_code}</td>
                    <td className="px-4 sm:px-6 py-4 font-medium text-gray-900">
                      <div>
                        <p className="font-mono text-xs">{row.patient_code || '—'}</p>
                        {row.patient_name && <p className="text-xs text-gray-500">{row.patient_name}</p>}
                      </div>
                    </td>
                    <td className="px-4 sm:px-6 py-4 text-gray-500 whitespace-nowrap text-xs">
                      {new Date(row.created_at).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' })}
                    </td>
                    <td className="px-4 sm:px-6 py-4">
                      <span className={`px-2.5 py-1 rounded-md text-xs font-medium border whitespace-nowrap ${findingsColor(row.findings_label)}`}>
                        {row.findings_label || 'Normal'}
                      </span>
                    </td>
                    <td className="px-4 sm:px-6 py-4 text-right whitespace-nowrap">
                      <button 
                        onClick={() => handleReportAction(row, 'view')}
                        className="text-blue-600 hover:text-blue-700 font-medium transition-colors mr-3 sm:mr-4 inline-flex items-center gap-1 text-xs"
                      >
                        <Eye size={13} /> View
                      </button>
                      <button
                        onClick={() => handleReportAction(row, 'download')}
                        className="text-gray-600 hover:text-gray-900 font-medium transition-colors inline-flex items-center gap-1 text-xs"
                      >
                        <Download size={13} /> Download
                      </button>
                    </td>
                  </motion.tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
