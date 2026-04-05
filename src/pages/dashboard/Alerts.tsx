import { useEffect, useState } from 'react';
import { motion } from 'motion/react';
import { AlertCircle, CheckCircle2, ShieldAlert, Flag, Clock } from 'lucide-react';
import { supabase } from '../../lib/supabase';

interface AlertRec {
  id: string;
  severity: 'critical' | 'high' | 'medium' | 'low';
  title: string;
  description: string;
  status: 'sent' | 'acknowledged' | 'reviewed';
  created_at: string;
  patient_code?: string;
  doctor_name?: string;
}

export default function Alerts() {
  const [alerts, setAlerts] = useState<AlertRec[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadAlerts();
  }, []);

  const loadAlerts = async () => {
    setLoading(true);
    try {
      const { data } = await supabase
        .from('v_active_alerts')
        .select('*')
        .order('created_at', { ascending: false });
      setAlerts(data || []);
    } catch (err) {
      console.error('Failed to load alerts:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleUpdateStatus = async (id: string, newStatus: 'acknowledged' | 'reviewed') => {
    const updateObj: any = { status: newStatus };
    if (newStatus === 'acknowledged') {
      updateObj.acknowledged_at = new Date().toISOString();
    } else if (newStatus === 'reviewed') {
      updateObj.reviewed_at = new Date().toISOString();
    }

    try {
      const { error } = await supabase.from('alerts').update(updateObj).eq('id', id);
      if (error) throw error;
      
      if (newStatus === 'reviewed') {
        // Remove from active view
        setAlerts(alerts.filter(a => a.id !== id));
      } else {
        // Update in-place
        setAlerts(alerts.map(a => a.id === id ? { ...a, status: newStatus } : a));
      }
    } catch (err) {
      console.error('Failed to update alert:', err);
    }
  };

  const getSeverityColors = (severity: string) => {
    if (severity === 'critical' || severity === 'high') return 'bg-red-50 text-red-700 border-red-200';
    if (severity === 'medium') return 'bg-amber-50 text-amber-700 border-amber-200';
    return 'bg-blue-50 text-blue-700 border-blue-200';
  };

  return (
    <div className="max-w-6xl mx-auto space-y-6 sm:space-y-8">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6 sm:mb-8">
        <div>
          <h1 className="text-xl sm:text-2xl font-semibold text-gray-900 tracking-tight mb-2">System Alerts</h1>
          <p className="text-gray-500 text-sm">
            Review critical findings and required actions from the FetalAI analysis pipelines.
          </p>
        </div>
      </div>

      <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
        <div className="divide-y divide-gray-100">
          {loading ? (
            <div className="p-12 text-center text-gray-400">Loading alerts...</div>
          ) : alerts.length === 0 ? (
            <div className="p-16 text-center flex flex-col items-center">
              <ShieldAlert size={48} className="text-gray-200 mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-1">No Active Alerts</h3>
              <p className="text-sm text-gray-500">All alerts have been reviewed or none have been generated yet.</p>
            </div>
          ) : (
            alerts.map((alert, index) => (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.05 }}
                key={alert.id}
                className="p-5 hover:bg-gray-50 transition-colors w-full"
              >
                <div className="flex flex-col sm:flex-row gap-4 sm:gap-6 justify-between items-start">
                  
                  {/* Left Side: Detail section */}
                  <div className="flex items-start gap-4 flex-grow">
                    <div className={`mt-1 w-10 h-10 rounded-full flex items-center justify-center shrink-0 border ${
                      alert.severity === 'critical' || alert.severity === 'high' 
                      ? 'bg-red-50 border-red-200 text-red-600' 
                      : 'bg-amber-50 border-amber-200 text-amber-600'
                    }`}>
                      <AlertCircle size={20} />
                    </div>
                    
                    <div className="space-y-2 w-full">
                      <div className="flex flex-col sm:flex-row sm:items-center gap-2 flex-wrap">
                        <h3 className="text-sm font-semibold text-gray-900">{alert.title}</h3>
                        <span className={`inline-flex items-center px-2 py-0.5 rounded text-[10px] font-medium border uppercase tracking-wider ${getSeverityColors(alert.severity)}`}>
                          {alert.severity}
                        </span>
                        {alert.status === 'acknowledged' && (
                          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-medium border bg-blue-50 text-blue-700 border-blue-200 uppercase tracking-wider">
                            <Clock size={10} /> Acknowledged
                          </span>
                        )}
                      </div>
                      
                      <p className="text-sm text-gray-700 py-1 bg-white p-3 rounded border border-gray-100 shadow-sm leading-relaxed">
                        {alert.description}
                      </p>
                      
                      <div className="flex flex-wrap items-center gap-4 text-xs text-gray-500 pt-1">
                        {alert.patient_code && (
                          <div className="flex items-center gap-1.5 font-medium">
                            <span className="text-gray-400">Patient:</span> {alert.patient_code}
                          </div>
                        )}
                        {alert.doctor_name && (
                          <div className="flex items-center gap-1.5 border-l border-gray-200 pl-4">
                            <span className="text-gray-400">Doctor:</span> {alert.doctor_name}
                          </div>
                        )}
                        <div className="flex items-center gap-1.5 border-l border-gray-200 pl-4">
                          <span className="text-gray-400">Timestamp:</span> 
                          {new Date(alert.created_at).toLocaleString()}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Right Side: Actions */}
                  <div className="shrink-0 flex sm:flex-col gap-2 w-full sm:w-auto mt-2 sm:mt-0 justify-end">
                    {alert.status === 'sent' && (
                      <button
                        onClick={() => handleUpdateStatus(alert.id, 'acknowledged')}
                        className="px-3 py-1.5 bg-white border border-gray-300 text-gray-700 rounded text-xs font-medium hover:bg-gray-50 hover:text-gray-900 transition-colors shadow-sm flex items-center justify-center gap-1.5 w-full sm:w-32 whitespace-nowrap"
                      >
                        <Flag size={14} /> Acknowledge
                      </button>
                    )}
                    <button
                      onClick={() => handleUpdateStatus(alert.id, 'reviewed')}
                      className="px-3 py-1.5 bg-[var(--color-primary)] text-white border border-transparent rounded text-xs font-medium hover:bg-blue-700 transition-colors shadow-sm flex items-center justify-center gap-1.5 w-full sm:w-32 whitespace-nowrap"
                    >
                      <CheckCircle2 size={14} /> Resolve
                    </button>
                  </div>

                </div>
              </motion.div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
