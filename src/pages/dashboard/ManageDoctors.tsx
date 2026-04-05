import React, { useState, useEffect } from 'react';
import { motion } from 'motion/react';
import { Plus, Search, UserPlus, Shield, Mail, Phone, X, CheckCircle2 } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { supabase } from '../../lib/supabase';
import { createDoctorAccount } from '../../lib/auth';

interface Doctor {
  id: string;
  full_name: string;
  email?: string;
  phone?: string;
  specialty?: string;
  department?: string;
  is_active: boolean;
  doctor_code?: string;
  joined_at: string;
}

export default function ManageDoctors() {
  const navigate = useNavigate();
  const [doctors, setDoctors] = useState<Doctor[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  const [selectedDoctor, setSelectedDoctor] = useState<Doctor | null>(null);
  const [saving, setSaving] = useState(false);
  const [toast, setToast] = useState<string | null>(null);

  const [newDoctor, setNewDoctor] = useState({
    full_name: '',
    email: '',
    password: '',
    phone: '',
    specialty: '',
    department: '',
  });

  useEffect(() => {
    const userRole = localStorage.getItem('userRole');
    if (userRole !== 'admin') { navigate('/dashboard'); return; }
    loadDoctors();
  }, [navigate]);

  const loadDoctors = async () => {
    setLoading(true);
    try {
      const { data } = await supabase
        .from('profiles')
        .select('id, full_name, phone, specialty, department, is_active, doctor_code, joined_at')
        .eq('role', 'doctor')
        .order('joined_at', { ascending: false });
      setDoctors(data || []);
    } catch (err) {
      console.error('Failed to load doctors:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleAddDoctor = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);
    try {
      await createDoctorAccount({
        email: newDoctor.email,
        password: newDoctor.password,
        full_name: newDoctor.full_name,
        specialty: newDoctor.specialty,
        department: newDoctor.department,
        phone: newDoctor.phone,
      });
      setIsAddModalOpen(false);
      setNewDoctor({ full_name: '', email: '', password: '', phone: '', specialty: '', department: '' });
      showToast('Doctor account created successfully');
      loadDoctors();
    } catch (err: unknown) {
      showToast(err instanceof Error ? err.message : 'Failed to create account');
    } finally {
      setSaving(false);
    }
  };

  const toggleDoctorStatus = async (doctor: Doctor) => {
    try {
      await supabase
        .from('profiles')
        .update({ is_active: !doctor.is_active })
        .eq('id', doctor.id);
      setDoctors(prev => prev.map(d => d.id === doctor.id ? { ...d, is_active: !d.is_active } : d));
      if (selectedDoctor?.id === doctor.id) setSelectedDoctor({ ...selectedDoctor, is_active: !doctor.is_active });
      showToast(`Doctor ${!doctor.is_active ? 'activated' : 'deactivated'} successfully`);
    } catch {
      showToast('Failed to update status');
    }
  };

  const showToast = (msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(null), 3000);
  };

  const filteredDoctors = doctors.filter(d =>
    (d.full_name || '').toLowerCase().includes(searchQuery.toLowerCase()) ||
    (d.doctor_code || '').toLowerCase().includes(searchQuery.toLowerCase()) ||
    (d.specialty || '').toLowerCase().includes(searchQuery.toLowerCase())
  );

  const inputCls = "w-full bg-white border border-gray-300 rounded-lg px-3.5 py-2.5 focus:outline-none focus:border-[var(--color-primary)] focus:ring-1 focus:ring-[var(--color-primary)] text-sm shadow-sm";

  return (
    <div className="max-w-6xl mx-auto space-y-6 sm:space-y-8">
      {/* Toast */}
      {toast && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed top-4 right-4 z-50 bg-gray-900 text-white text-sm px-4 py-2.5 rounded-lg shadow-lg flex items-center gap-2"
        >
          <CheckCircle2 size={16} className="text-emerald-400" />
          {toast}
        </motion.div>
      )}

      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6 sm:mb-8">
        <div>
          <h1 className="text-xl sm:text-2xl font-semibold text-gray-900 tracking-tight mb-2">Manage Doctors</h1>
          <p className="text-gray-500 text-sm">View and manage medical personnel access to the FetalAI system.</p>
        </div>
        <button
          onClick={() => setIsAddModalOpen(true)}
          className="bg-[var(--color-primary)] text-white px-4 py-2.5 rounded-lg text-sm font-medium hover:bg-[var(--color-primary-light)] transition-colors flex items-center gap-2 shadow-sm whitespace-nowrap"
        >
          <UserPlus size={18} />
          Add New Doctor
        </button>
      </div>

      {/* Search */}
      <div className="bg-white p-4 rounded-xl border border-gray-200 shadow-sm flex flex-col sm:flex-row gap-4">
        <div className="relative flex-grow">
          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-gray-400">
            <Search size={18} />
          </div>
          <input
            type="text"
            className="w-full bg-gray-50 border border-gray-200 rounded-lg pl-10 pr-4 py-2.5 focus:outline-none focus:border-[var(--color-primary)] focus:ring-1 focus:ring-[var(--color-primary)] transition-all text-sm"
            placeholder="Search by name, code, or specialty..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
      </div>

      {/* Table */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="bg-gray-50 border-b border-gray-200 text-xs uppercase tracking-wider text-gray-500 font-semibold">
                <th className="px-6 py-4">Doctor</th>
                <th className="px-6 py-4">Contact</th>
                <th className="px-6 py-4">Department</th>
                <th className="px-6 py-4">Status</th>
                <th className="px-6 py-4 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {loading ? (
                <tr>
                  <td colSpan={5} className="px-6 py-12 text-center text-gray-400">Loading doctors...</td>
                </tr>
              ) : filteredDoctors.length === 0 ? (
                <tr>
                  <td colSpan={5} className="px-6 py-12 text-center text-gray-500">
                    {doctors.length === 0 ? 'No doctors added yet.' : 'No doctors match your search.'}
                  </td>
                </tr>
              ) : (
                filteredDoctors.map((doctor, index) => (
                  <motion.tr
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.05 }}
                    key={doctor.id}
                    className="hover:bg-gray-50 transition-colors group cursor-pointer"
                    onClick={() => setSelectedDoctor(doctor)}
                  >
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-full bg-[var(--color-primary)]/10 text-[var(--color-primary)] flex items-center justify-center font-medium text-sm">
                          {(doctor.full_name || 'D').split(' ').map(n => n[0]).join('').substring(0, 2).toUpperCase()}
                        </div>
                        <div>
                          <p className="text-sm font-medium text-gray-900">{doctor.full_name}</p>
                          <p className="text-xs text-gray-500 font-mono">{doctor.doctor_code || doctor.id.substring(0, 8)}</p>
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <div className="space-y-1 text-xs text-gray-600">
                        {doctor.phone && <div className="flex items-center gap-2"><Phone size={12} className="text-gray-400" />{doctor.phone}</div>}
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <p className="text-sm text-gray-900">{doctor.department || '—'}</p>
                      <p className="text-xs text-gray-500">{doctor.specialty || '—'}</p>
                    </td>
                    <td className="px-6 py-4">
                      <span className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium ${doctor.is_active ? 'bg-emerald-50 text-emerald-700 border border-emerald-200' : 'bg-gray-100 text-gray-700 border border-gray-200'}`}>
                        {doctor.is_active ? 'Active' : 'Inactive'}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-right">
                      <button
                        onClick={(e) => { e.stopPropagation(); toggleDoctorStatus(doctor); }}
                        className={`text-xs font-medium px-3 py-1.5 rounded-lg transition-colors ${doctor.is_active ? 'text-red-600 hover:bg-red-50' : 'text-emerald-600 hover:bg-emerald-50'}`}
                      >
                        {doctor.is_active ? 'Deactivate' : 'Activate'}
                      </button>
                    </td>
                  </motion.tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Doctor Detail Modal */}
      {selectedDoctor && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-gray-900/50 backdrop-blur-sm">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-white rounded-2xl shadow-xl w-full max-w-md overflow-hidden"
          >
            <div className="px-6 py-4 border-b border-gray-100 flex items-center justify-between bg-gray-50/50">
              <h2 className="text-lg font-semibold text-gray-900">Doctor Profile</h2>
              <button onClick={() => setSelectedDoctor(null)} className="text-gray-400 hover:text-gray-600 transition-colors">
                <X size={20} />
              </button>
            </div>
            <div className="p-6">
              <div className="flex items-center gap-4 mb-6">
                <div className="w-16 h-16 rounded-full bg-[var(--color-primary)]/10 text-[var(--color-primary)] flex items-center justify-center font-medium text-xl">
                  {(selectedDoctor.full_name || 'D').split(' ').map(n => n[0]).join('').substring(0, 2).toUpperCase()}
                </div>
                <div>
                  <h3 className="text-xl font-semibold text-gray-900">{selectedDoctor.full_name}</h3>
                  <p className="text-sm text-gray-500 font-mono">{selectedDoctor.doctor_code || selectedDoctor.id.substring(0, 8)}</p>
                </div>
              </div>
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-gray-50 p-3 rounded-lg border border-gray-100">
                    <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-1">Status</p>
                    <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${selectedDoctor.is_active ? 'bg-emerald-100 text-emerald-800' : 'bg-gray-200 text-gray-800'}`}>
                      {selectedDoctor.is_active ? 'Active' : 'Inactive'}
                    </span>
                  </div>
                  <div className="bg-gray-50 p-3 rounded-lg border border-gray-100">
                    <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-1">Joined</p>
                    <p className="text-sm text-gray-900 font-medium">
                      {new Date(selectedDoctor.joined_at).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' })}
                    </p>
                  </div>
                </div>
                <div className="bg-gray-50 p-4 rounded-lg border border-gray-100 space-y-3">
                  <div>
                    <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-1">Department & Specialty</p>
                    <p className="text-sm text-gray-900 font-medium">{selectedDoctor.department || '—'}</p>
                    <p className="text-sm text-gray-600">{selectedDoctor.specialty || '—'}</p>
                  </div>
                  {selectedDoctor.phone && (
                    <div className="pt-3 border-t border-gray-200">
                      <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Contact</p>
                      <div className="flex items-center gap-2 text-sm text-gray-700">
                        <Phone size={14} className="text-gray-400" />
                        {selectedDoctor.phone}
                      </div>
                    </div>
                  )}
                </div>
              </div>
              <div className="mt-6 pt-6 border-t border-gray-100 flex justify-end gap-3">
                <button
                  onClick={() => toggleDoctorStatus(selectedDoctor)}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${selectedDoctor.is_active ? 'bg-red-50 text-red-600 hover:bg-red-100' : 'bg-emerald-50 text-emerald-600 hover:bg-emerald-100'}`}
                >
                  {selectedDoctor.is_active ? 'Deactivate Account' : 'Activate Account'}
                </button>
              </div>
            </div>
          </motion.div>
        </div>
      )}

      {/* Add Doctor Modal */}
      {isAddModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-gray-900/50 backdrop-blur-sm">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-white rounded-2xl shadow-xl w-full max-w-lg overflow-hidden"
          >
            <div className="px-6 py-4 border-b border-gray-100 flex items-center justify-between bg-gray-50/50">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg bg-[var(--color-primary)]/10 text-[var(--color-primary)] flex items-center justify-center">
                  <Shield size={18} />
                </div>
                <h2 className="text-lg font-semibold text-gray-900">Add New Doctor</h2>
              </div>
              <button onClick={() => setIsAddModalOpen(false)} className="text-gray-400 hover:text-gray-600 transition-colors">
                <X size={20} />
              </button>
            </div>
            <form onSubmit={handleAddDoctor} className="p-6 space-y-4">
              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-gray-600 uppercase tracking-wider">Full Name</label>
                <input required type="text" value={newDoctor.full_name} onChange={e => setNewDoctor({ ...newDoctor, full_name: e.target.value })} className={inputCls} placeholder="Dr. Jane Doe" />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <label className="text-xs font-semibold text-gray-600 uppercase tracking-wider">Email</label>
                  <input required type="email" value={newDoctor.email} onChange={e => setNewDoctor({ ...newDoctor, email: e.target.value })} className={inputCls} placeholder="doctor@fetalai.com" />
                </div>
                <div className="space-y-1.5">
                  <label className="text-xs font-semibold text-gray-600 uppercase tracking-wider">Password</label>
                  <input required type="password" value={newDoctor.password} onChange={e => setNewDoctor({ ...newDoctor, password: e.target.value })} className={inputCls} placeholder="Min 8 characters" minLength={8} />
                </div>
              </div>
              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-gray-600 uppercase tracking-wider">Phone</label>
                <input type="tel" value={newDoctor.phone} onChange={e => setNewDoctor({ ...newDoctor, phone: e.target.value })} className={inputCls} placeholder="+92 300 0000000" />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <label className="text-xs font-semibold text-gray-600 uppercase tracking-wider">Department</label>
                  <select required value={newDoctor.department} onChange={e => setNewDoctor({ ...newDoctor, department: e.target.value })} className={inputCls}>
                    <option value="">Select Department</option>
                    <option>Obstetrics & Gynecology</option>
                    <option>Cardiology</option>
                    <option>Imaging</option>
                    <option>Pediatrics</option>
                    <option>Fetal Medicine</option>
                  </select>
                </div>
                <div className="space-y-1.5">
                  <label className="text-xs font-semibold text-gray-600 uppercase tracking-wider">Specialty</label>
                  <input required type="text" value={newDoctor.specialty} onChange={e => setNewDoctor({ ...newDoctor, specialty: e.target.value })} className={inputCls} placeholder="e.g. Fetal Medicine Specialist" />
                </div>
              </div>
              <div className="pt-4 flex gap-3">
                <button type="button" onClick={() => setIsAddModalOpen(false)} className="flex-1 px-4 py-2.5 bg-white border border-gray-300 text-gray-700 rounded-lg text-sm font-medium hover:bg-gray-50 transition-colors shadow-sm">
                  Cancel
                </button>
                <button type="submit" disabled={saving} className="flex-1 px-4 py-2.5 bg-[var(--color-primary)] text-white rounded-lg text-sm font-medium hover:bg-[var(--color-primary-light)] transition-colors shadow-sm disabled:opacity-60">
                  {saving ? 'Creating...' : 'Create Account'}
                </button>
              </div>
            </form>
          </motion.div>
        </div>
      )}
    </div>
  );
}
