import React, { useState } from 'react';
import { motion } from 'motion/react';
import { Search, Calendar, FileText, ImageIcon, ChevronRight, Filter } from 'lucide-react';

interface AnalysisRecord {
  id: string;
  patientId: string;
  date: string;
  type: 'VLM' | 'LLM' | 'Combined';
  status: 'Completed' | 'Pending' | 'Flagged';
  summary: string;
}

const mockHistory: AnalysisRecord[] = [
  {
    id: 'ANL-2024-001',
    patientId: 'PT-8842',
    date: '2024-04-05T10:30:00Z',
    type: 'Combined',
    status: 'Completed',
    summary: 'Normal fetal growth parameters. No anomalies detected in ultrasound or clinical data.'
  },
  {
    id: 'ANL-2024-002',
    patientId: 'PT-9105',
    date: '2024-04-04T14:15:00Z',
    type: 'VLM',
    status: 'Flagged',
    summary: 'Potential irregularity in nuchal translucency measurement. Recommend follow-up.'
  },
  {
    id: 'ANL-2024-003',
    patientId: 'PT-7731',
    date: '2024-04-03T09:45:00Z',
    type: 'LLM',
    status: 'Completed',
    summary: 'Maternal blood pressure trends indicate mild gestational hypertension. Monitor closely.'
  },
  {
    id: 'ANL-2024-004',
    patientId: 'PT-8842',
    date: '2024-03-28T11:20:00Z',
    type: 'VLM',
    status: 'Completed',
    summary: 'Routine anatomy scan. All major organ systems appear normal.'
  }
];

export default function AnalysisHistory() {
  const [searchQuery, setSearchQuery] = useState('');
  const [filterType, setFilterType] = useState('All');

  const filteredHistory = mockHistory.filter(record => {
    const matchesSearch = record.patientId.toLowerCase().includes(searchQuery.toLowerCase()) || 
                          record.id.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesType = filterType === 'All' || record.type === filterType;
    return matchesSearch && matchesType;
  });

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'VLM': return <ImageIcon size={16} className="text-blue-500" />;
      case 'LLM': return <FileText size={16} className="text-emerald-500" />;
      case 'Combined': return <div className="flex -space-x-1"><ImageIcon size={16} className="text-blue-500" /><FileText size={16} className="text-emerald-500 bg-white rounded-full" /></div>;
      default: return <FileText size={16} />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'Completed': return 'bg-emerald-50 text-emerald-700 border-emerald-200';
      case 'Flagged': return 'bg-amber-50 text-amber-700 border-amber-200';
      case 'Pending': return 'bg-blue-50 text-blue-700 border-blue-200';
      default: return 'bg-gray-50 text-gray-700 border-gray-200';
    }
  };

  return (
    <div className="max-w-6xl mx-auto space-y-6 sm:space-y-8">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6 sm:mb-8">
        <div>
          <h1 className="text-xl sm:text-2xl font-semibold text-gray-900 tracking-tight mb-2">Analysis History</h1>
          <p className="text-gray-500 text-sm">
            Review past VLM and LLM analyses, patient records, and AI-generated insights.
          </p>
        </div>
      </div>

      {/* Search and Filters */}
      <div className="bg-white p-4 rounded-xl border border-gray-200 shadow-sm flex flex-col sm:flex-row gap-4">
        <div className="relative flex-grow">
          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-gray-400">
            <Search size={18} />
          </div>
          <input
            type="text"
            className="w-full bg-gray-50 border border-gray-200 rounded-lg pl-10 pr-4 py-2.5 focus:outline-none focus:border-[var(--color-primary)] focus:ring-1 focus:ring-[var(--color-primary)] transition-all text-sm"
            placeholder="Search by Patient ID or Analysis ID..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
        <div className="flex gap-2">
          <div className="relative">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-gray-400">
              <Filter size={16} />
            </div>
            <select 
              value={filterType}
              onChange={(e) => setFilterType(e.target.value)}
              className="bg-gray-50 border border-gray-200 rounded-lg pl-9 pr-8 py-2.5 focus:outline-none focus:border-[var(--color-primary)] focus:ring-1 focus:ring-[var(--color-primary)] transition-all text-sm text-gray-700 appearance-none"
            >
              <option value="All">All Types</option>
              <option value="VLM">VLM (Image)</option>
              <option value="LLM">LLM (Clinical)</option>
              <option value="Combined">Combined</option>
            </select>
          </div>
        </div>
      </div>

      {/* History List */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
        <div className="divide-y divide-gray-100">
          {filteredHistory.map((record, index) => (
            <motion.div 
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.05 }}
              key={record.id} 
              className="p-5 hover:bg-gray-50 transition-colors cursor-pointer group flex flex-col sm:flex-row gap-4 sm:items-center justify-between"
            >
              <div className="flex items-start gap-4 flex-grow">
                <div className="mt-1 w-10 h-10 rounded-full bg-gray-100 flex items-center justify-center shrink-0 border border-gray-200">
                  {getTypeIcon(record.type)}
                </div>
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <h3 className="text-sm font-semibold text-gray-900">{record.patientId}</h3>
                    <span className="text-gray-300">•</span>
                    <span className="text-xs font-mono text-gray-500">{record.id}</span>
                    <span className={`ml-2 inline-flex items-center px-2 py-0.5 rounded text-[10px] font-medium border ${getStatusColor(record.status)}`}>
                      {record.status}
                    </span>
                  </div>
                  <p className="text-sm text-gray-600 line-clamp-2 sm:line-clamp-1">{record.summary}</p>
                  <div className="flex items-center gap-4 text-xs text-gray-500 pt-1">
                    <div className="flex items-center gap-1.5">
                      <Calendar size={12} />
                      {new Date(record.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
                    </div>
                    <div className="flex items-center gap-1.5">
                      <span className="font-medium text-gray-700">Type:</span> {record.type}
                    </div>
                  </div>
                </div>
              </div>
              <div className="shrink-0 flex items-center justify-end sm:justify-center">
                <button className="text-sm font-medium text-[var(--color-primary)] hover:text-[var(--color-primary-light)] flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                  View Details
                  <ChevronRight size={16} />
                </button>
              </div>
            </motion.div>
          ))}
          {filteredHistory.length === 0 && (
            <div className="p-12 text-center text-gray-500">
              No analysis records found matching your criteria.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
