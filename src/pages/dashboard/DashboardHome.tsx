import { motion } from 'motion/react';
import { Activity, AlertCircle, ArrowRight, Clock, Plus, FileText, CheckCircle2 } from 'lucide-react';
import { Link } from 'react-router-dom';

export default function DashboardHome() {
  return (
    <div className="max-w-6xl mx-auto space-y-8">
      {/* Header / Greeting */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
          <h1 className="text-2xl font-semibold text-gray-900 tracking-tight">Overview</h1>
          <p className="text-gray-500 text-sm mt-1">
            Welcome back, Dr. Jenkins. Here's what's happening today.
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

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {[
          { label: 'Total Scans Analyzed', value: '1,284', icon: <Activity size={20} />, color: 'text-blue-600', bg: 'bg-blue-50', border: 'border-blue-100' },
          { label: 'Pending Reports', value: '12', icon: <Clock size={20} />, color: 'text-amber-600', bg: 'bg-amber-50', border: 'border-amber-100' },
          { label: 'Critical Alerts', value: '2', icon: <AlertCircle size={20} />, color: 'text-red-600', bg: 'bg-red-50', border: 'border-red-100' },
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
        {/* Recent Analyses Table */}
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
            <Link to="/dashboard/reports" className="text-sm text-blue-600 hover:text-blue-700 font-medium transition-colors">
              View All
            </Link>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-white text-xs uppercase tracking-wider text-gray-500 border-b border-gray-100">
                  <th className="px-5 py-3 font-medium">Patient ID</th>
                  <th className="px-5 py-3 font-medium">Date</th>
                  <th className="px-5 py-3 font-medium">Status</th>
                  <th className="px-5 py-3 font-medium text-right">Action</th>
                </tr>
              </thead>
              <tbody className="text-sm divide-y divide-gray-50">
                {[
                  { id: 'PT-8921', date: 'Oct 24, 2023', status: 'Complete', statusColor: 'text-emerald-700 bg-emerald-50 border-emerald-200' },
                  { id: 'PT-8922', date: 'Oct 24, 2023', status: 'Review Required', statusColor: 'text-amber-700 bg-amber-50 border-amber-200' },
                  { id: 'PT-8923', date: 'Oct 23, 2023', status: 'Complete', statusColor: 'text-emerald-700 bg-emerald-50 border-emerald-200' },
                  { id: 'PT-8924', date: 'Oct 23, 2023', status: 'Processing', statusColor: 'text-blue-700 bg-blue-50 border-blue-200' },
                  { id: 'PT-8925', date: 'Oct 22, 2023', status: 'Complete', statusColor: 'text-emerald-700 bg-emerald-50 border-emerald-200' },
                ].map((row, idx) => (
                  <tr key={idx} className="hover:bg-gray-50/50 transition-colors">
                    <td className="px-5 py-3.5 font-medium text-gray-900">{row.id}</td>
                    <td className="px-5 py-3.5 text-gray-500">{row.date}</td>
                    <td className="px-5 py-3.5">
                      <span className={`px-2.5 py-1 rounded-md text-xs font-medium border ${row.statusColor}`}>
                        {row.status}
                      </span>
                    </td>
                    <td className="px-5 py-3.5 text-right">
                      <button className="text-gray-400 hover:text-gray-900 transition-colors">
                        <ArrowRight size={16} />
                      </button>
                    </td>
                  </tr>
                ))}
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
              System Alerts
            </h3>
          </div>
          <div className="p-5 space-y-4 flex-grow">
            {[
              { title: 'High Risk Detected', desc: 'Anomaly detected in PT-8922 scan.', time: '2 hours ago', type: 'critical' },
              { title: 'System Update', desc: 'AI model updated to v2.4.1.', time: '1 day ago', type: 'info' },
              { title: 'Data Sync Complete', desc: 'Clinical records synced successfully.', time: '2 days ago', type: 'success' },
            ].map((alert, idx) => (
              <div key={idx} className="flex gap-3">
                <div className="mt-0.5">
                  {alert.type === 'critical' ? (
                    <AlertCircle size={16} className="text-red-500" />
                  ) : alert.type === 'success' ? (
                    <CheckCircle2 size={16} className="text-emerald-500" />
                  ) : (
                    <Activity size={16} className="text-blue-500" />
                  )}
                </div>
                <div>
                  <h4 className="text-sm font-medium text-gray-900">{alert.title}</h4>
                  <p className="text-xs text-gray-500 mt-0.5">{alert.desc}</p>
                  <p className="text-[10px] text-gray-400 mt-1.5 uppercase tracking-wider font-medium">{alert.time}</p>
                </div>
              </div>
            ))}
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

