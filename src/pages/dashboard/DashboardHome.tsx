import { useEffect, useState } from 'react';
import { motion } from 'motion/react';
import { Activity, AlertCircle, ArrowRight, Clock, Plus, FileText, CheckCircle2 } from 'lucide-react';
import { Link } from 'react-router-dom';
import { supabase } from '../../lib/supabase';

interface DashboardStats {
  totalAnalyses: number;
  pendingReports: number;
  criticalAlerts: number;
}

interface RecentAnalysis {
  id: string;
  patient_code: string;
  status: string;
  created_at: string;
  overall_risk_level: string;
}

interface AlertItem {
  id: string;
  title: string;
  description: string;
  severity: string;
  created_at: string;
}

export default function DashboardHome() {
  const [stats, setStats] = useState<DashboardStats>({ totalAnalyses: 0, pendingReports: 0, criticalAlerts: 0 });
  const [recentAnalyses, setRecentAnalyses] = useState<RecentAnalysis[]>([]);
  const [alerts, setAlerts] = useState<AlertItem[]>([]);
  const [loading, setLoading] = useState(true);
  const userName = localStorage.getItem('userName') || 'Doctor';

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    setLoading(true);
    try {
      // Fetch stats
      const [analysesRes, reportsRes, alertsRes] = await Promise.all([
        supabase.from('analysis_results').select('id', { count: 'exact', head: true }),
        supabase.from('reports').select('id', { count: 'exact', head: true }).eq('is_finalized', false),
        supabase.from('alerts').select('id', { count: 'exact', head: true }).eq('status', 'sent'),
      ]);

      setStats({
        totalAnalyses: analysesRes.count || 0,
        pendingReports: reportsRes.count || 0,
        criticalAlerts: alertsRes.count || 0,
      });

      // Fetch recent analyses
      const { data: recent } = await supabase
        .from('v_recent_analyses')
        .select('id, patient_code, status, created_at, overall_risk_level')
        .limit(5);

      setRecentAnalyses(recent || []);

      // Fetch active alerts
      const { data: activeAlerts } = await supabase
        .from('v_active_alerts')
        .select('id, title, description, severity, created_at')
        .limit(3);

      setAlerts(activeAlerts || []);
    } catch (err) {
      console.error('Dashboard load error:', err);
    } finally {
      setLoading(false);
    }
  };

  const formatDate = (iso: string) =>
    new Date(iso).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' });

  const timeAgo = (iso: string) => {
    const diff = Date.now() - new Date(iso).getTime();
    const hours = Math.floor(diff / 3_600_000);
    if (hours < 1) return 'Just now';
    if (hours < 24) return `${hours}h ago`;
    return `${Math.floor(hours / 24)}d ago`;
  };

  const statusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'text-emerald-700 bg-emerald-50 border-emerald-200';
      case 'failed': return 'text-red-700 bg-red-50 border-red-200';
      case 'in_progress': return 'text-blue-700 bg-blue-50 border-blue-200';
      default: return 'text-amber-700 bg-amber-50 border-amber-200';
    }
  };

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
          <h1 className="text-2xl font-semibold text-gray-900 tracking-tight">Overview</h1>
          <p className="text-gray-500 text-sm mt-1">
            Welcome back, {userName}. Here's what's happening today.
          </p>
        </motion.div>
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
          <Link
            to="/dashboard/analyze"
            className="px-5 py-2.5 bg-gray-900 text-white rounded-lg text-sm font-medium hover:bg-gray-800 transition-colors inline-flex items-center gap-2 shadow-sm"
          >
            <Plus size={16} /> New Analysis
          </Link>
        </motion.div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {[
          { label: 'Total Scans Analysed', value: loading ? '—' : stats.totalAnalyses.toLocaleString(), icon: <Activity size={20} />, color: 'text-blue-600', bg: 'bg-blue-50', border: 'border-blue-100' },
          { label: 'Pending Reports', value: loading ? '—' : stats.pendingReports.toString(), icon: <Clock size={20} />, color: 'text-amber-600', bg: 'bg-amber-50', border: 'border-amber-100' },
          { label: 'Critical Alerts', value: loading ? '—' : stats.criticalAlerts.toString(), icon: <AlertCircle size={20} />, color: 'text-red-600', bg: 'bg-red-50', border: 'border-red-100' },
        ].map((stat, idx) => (
          <motion.div
            key={idx}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: idx * 0.1 + 0.1 }}
            className="bg-white p-5 rounded-xl border border-gray-200 shadow-sm flex items-center gap-5"
          >
            <div className={`w-12 h-12 rounded-lg ${stat.bg} ${stat.color} border ${stat.border} flex items-center justify-center flex-shrink-0`}>
              {stat.icon}
            </div>
            <div>
              <p className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-1">{stat.label}</p>
              <p className="text-2xl font-semibold text-gray-900 tracking-tight">{stat.value}</p>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Recent Analyses */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="lg:col-span-2 bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden flex flex-col"
        >
          <div className="p-5 border-b border-gray-100 flex justify-between items-center bg-gray-50/50">
            <h3 className="font-semibold text-gray-900 flex items-center gap-2">
              <FileText size={16} className="text-gray-400" />
              Recent Analyses
            </h3>
            <Link to="/dashboard/history" className="text-sm text-blue-600 hover:text-blue-700 font-medium transition-colors">
              View All
            </Link>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-white text-xs uppercase tracking-wider text-gray-500 border-b border-gray-100">
                  <th className="px-5 py-3 font-medium">Patient</th>
                  <th className="px-5 py-3 font-medium">Date</th>
                  <th className="px-5 py-3 font-medium">Status</th>
                  <th className="px-5 py-3 font-medium text-right">Action</th>
                </tr>
              </thead>
              <tbody className="text-sm divide-y divide-gray-50">
                {recentAnalyses.length === 0 ? (
                  <tr>
                    <td colSpan={4} className="px-5 py-8 text-center text-gray-400 text-sm">
                      {loading ? 'Loading...' : 'No analyses yet. Upload a scan to get started.'}
                    </td>
                  </tr>
                ) : (
                  recentAnalyses.map((row) => (
                    <tr key={row.id} className="hover:bg-gray-50/50 transition-colors">
                      <td className="px-5 py-3.5 font-medium text-gray-900 font-mono text-xs">{row.patient_code || '—'}</td>
                      <td className="px-5 py-3.5 text-gray-500">{formatDate(row.created_at)}</td>
                      <td className="px-5 py-3.5">
                        <span className={`px-2.5 py-1 rounded-md text-xs font-medium border ${statusColor(row.status)}`}>
                          {row.status.replace('_', ' ')}
                        </span>
                      </td>
                      <td className="px-5 py-3.5 text-right">
                        <Link to="/dashboard/history" className="text-gray-400 hover:text-gray-900 transition-colors">
                          <ArrowRight size={16} />
                        </Link>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </motion.div>

        {/* System Alerts */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="bg-white rounded-xl border border-gray-200 shadow-sm flex flex-col"
        >
          <div className="p-5 border-b border-gray-100 bg-gray-50/50">
            <h3 className="font-semibold text-gray-900 flex items-center gap-2">
              <AlertCircle size={16} className="text-gray-400" />
              Active Alerts
            </h3>
          </div>
          <div className="p-5 space-y-4 flex-grow">
            {alerts.length === 0 ? (
              <div className="text-center text-gray-400 text-sm py-6">
                {loading ? 'Loading...' : (
                  <div className="flex flex-col items-center gap-2">
                    <CheckCircle2 size={20} className="text-emerald-400" />
                    No active alerts
                  </div>
                )}
              </div>
            ) : (
              alerts.map((alert) => (
                <div key={alert.id} className="flex gap-3">
                  <div className="mt-0.5">
                    <AlertCircle size={16} className={alert.severity === 'critical' || alert.severity === 'high' ? 'text-red-500' : 'text-amber-500'} />
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-gray-900">{alert.title}</h4>
                    <p className="text-xs text-gray-500 mt-0.5 line-clamp-2">{alert.description}</p>
                    <p className="text-[10px] text-gray-400 mt-1.5 uppercase tracking-wider font-medium">{timeAgo(alert.created_at)}</p>
                  </div>
                </div>
              ))
            )}
          </div>
          <div className="p-4 border-t border-gray-100">
            <Link to="/dashboard/alerts" className="block text-center w-full text-sm text-blue-600 hover:text-blue-700 font-medium transition-colors">
              View All Alerts
            </Link>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
