import React, { useState, useEffect } from 'react';
import { motion } from 'motion/react';
import { Plus, Search, MoreVertical, UserPlus, Shield, Mail, Phone, MapPin, X, AlertTriangle, CheckCircle2 } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

interface Doctor {
  id: string;
  name: string;
  email: string;
  phone: string;
  specialty: string;
  department: string;
  status: 'Active' | 'Inactive';
  joinDate: string;
}

const initialDoctors: Doctor[] = [
  {
    id: 'DOC-0000',
    name: 'Dr. Sarah Jenkins',
    email: 's.jenkins@fetalai.com',
    phone: '+1 (555) 123-4567',
    specialty: 'Fetal Medicine Specialist',
    department: 'Obstetrics & Gynecology',
    status: 'Active',
    joinDate: '2024-01-15'
  },
  {
    id: 'DOC-1024',
    name: 'Dr. Michael Chen',
    email: 'm.chen@fetalai.com',
    phone: '+1 (555) 987-6543',
    specialty: 'Pediatric Cardiologist',
    department: 'Cardiology',
    status: 'Active',
    joinDate: '2024-03-22'
  },
  {
    id: 'DOC-2048',
    name: 'Dr. Emily Rodriguez',
    email: 'e.rodriguez@fetalai.com',
    phone: '+1 (555) 456-7890',
    specialty: 'Radiologist',
    department: 'Imaging',
    status: 'Inactive',
    joinDate: '2023-11-05'
  }
];

export default function ManageDoctors() {
  const navigate = useNavigate();
  const [doctors, setDoctors] = useState<Doctor[]>(initialDoctors);
  const [searchQuery, setSearchQuery] = useState('');
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  const [selectedDoctor, setSelectedDoctor] = useState<Doctor | null>(null);
  const [isDetailsModalOpen, setIsDetailsModalOpen] = useState(false);

  useEffect(() => {
    const userRole = localStorage.getItem('userRole');
    if (userRole !== 'admin') {
      navigate('/dashboard');
    }
  }, [navigate]);

  const [newDoctor, setNewDoctor] = useState({
    name: '',
    email: '',
    phone: '',
    specialty: '',
    department: ''
  });

  const handleAddDoctor = (e: React.FormEvent) => {
    e.preventDefault();
    const id = `DOC-${Math.floor(1000 + Math.random() * 9000)}`;
    const doctor: Doctor = {
      id,
      ...newDoctor,
      status: 'Active',
      joinDate: new Date().toISOString().split('T')[0]
    };
    setDoctors([doctor, ...doctors]);
    setIsAddModalOpen(false);
    setNewDoctor({ name: '', email: '', phone: '', specialty: '', department: '' });
  };

  const toggleDoctorStatus = (id: string) => {
    setDoctors(doctors.map(doc => {
      if (doc.id === id) {
        return { ...doc, status: doc.status === 'Active' ? 'Inactive' : 'Active' };
      }
      return doc;
    }));
    if (selectedDoctor && selectedDoctor.id === id) {
      setSelectedDoctor({ ...selectedDoctor, status: selectedDoctor.status === 'Active' ? 'Inactive' : 'Active' });
    }
  };

  const openDetails = (doctor: Doctor) => {
    setSelectedDoctor(doctor);
    setIsDetailsModalOpen(true);
  };

  const filteredDoctors = doctors.filter(doc => 
    doc.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    doc.id.toLowerCase().includes(searchQuery.toLowerCase()) ||
    doc.specialty.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="max-w-6xl mx-auto space-y-6 sm:space-y-8">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6 sm:mb-8">
        <div>
          <h1 className="text-xl sm:text-2xl font-semibold text-gray-900 tracking-tight mb-2">Manage Doctors</h1>
          <p className="text-gray-500 text-sm">
            View and manage medical personnel access to the FetalAI system.
          </p>
        </div>
        <button 
          onClick={() => setIsAddModalOpen(true)}
          className="bg-[var(--color-primary)] text-white px-4 py-2.5 rounded-lg text-sm font-medium hover:bg-[var(--color-primary-light)] transition-colors flex items-center gap-2 shadow-sm whitespace-nowrap"
        >
          <UserPlus size={18} />
          Add New Doctor
        </button>
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
            placeholder="Search by name, ID, or specialty..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
        <div className="flex gap-2">
          <select className="bg-gray-50 border border-gray-200 rounded-lg px-4 py-2.5 focus:outline-none focus:border-[var(--color-primary)] focus:ring-1 focus:ring-[var(--color-primary)] transition-all text-sm text-gray-700">
            <option value="all">All Departments</option>
            <option value="obgyn">Obstetrics & Gynecology</option>
            <option value="cardiology">Cardiology</option>
            <option value="imaging">Imaging</option>
          </select>
          <select className="bg-gray-50 border border-gray-200 rounded-lg px-4 py-2.5 focus:outline-none focus:border-[var(--color-primary)] focus:ring-1 focus:ring-[var(--color-primary)] transition-all text-sm text-gray-700">
            <option value="all">All Statuses</option>
            <option value="active">Active</option>
            <option value="inactive">Inactive</option>
          </select>
        </div>
      </div>

      {/* Doctors List */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="bg-gray-50 border-b border-gray-200 text-xs uppercase tracking-wider text-gray-500 font-semibold">
                <th className="px-6 py-4">Doctor</th>
                <th className="px-6 py-4">Contact Info</th>
                <th className="px-6 py-4">Department</th>
                <th className="px-6 py-4">Status</th>
                <th className="px-6 py-4 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {filteredDoctors.map((doctor, index) => (
                <motion.tr 
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.05 }}
                  key={doctor.id} 
                  className="hover:bg-gray-50 transition-colors group cursor-pointer"
                  onClick={() => openDetails(doctor)}
                >
                  <td className="px-6 py-4">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 rounded-full bg-[var(--color-primary)]/10 text-[var(--color-primary)] flex items-center justify-center font-medium text-sm">
                        {doctor.name.split(' ').map(n => n[0]).join('').substring(0, 2).toUpperCase()}
                      </div>
                      <div>
                        <p className="text-sm font-medium text-gray-900">{doctor.name}</p>
                        <p className="text-xs text-gray-500 font-mono">{doctor.id}</p>
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-4">
                    <div className="space-y-1">
                      <div className="flex items-center gap-2 text-xs text-gray-600">
                        <Mail size={12} className="text-gray-400" />
                        {doctor.email}
                      </div>
                      <div className="flex items-center gap-2 text-xs text-gray-600">
                        <Phone size={12} className="text-gray-400" />
                        {doctor.phone}
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-4">
                    <p className="text-sm text-gray-900">{doctor.department}</p>
                    <p className="text-xs text-gray-500">{doctor.specialty}</p>
                  </td>
                  <td className="px-6 py-4">
                    <span className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium ${
                      doctor.status === 'Active' 
                        ? 'bg-emerald-50 text-emerald-700 border border-emerald-200' 
                        : 'bg-gray-100 text-gray-700 border border-gray-200'
                    }`}>
                      {doctor.status}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-right">
                    <button 
                      onClick={(e) => {
                        e.stopPropagation();
                        toggleDoctorStatus(doctor.id);
                      }}
                      className={`text-xs font-medium px-3 py-1.5 rounded-lg transition-colors ${
                        doctor.status === 'Active' 
                          ? 'text-red-600 hover:bg-red-50' 
                          : 'text-emerald-600 hover:bg-emerald-50'
                      }`}
                    >
                      {doctor.status === 'Active' ? 'Deactivate' : 'Activate'}
                    </button>
                  </td>
                </motion.tr>
              ))}
              {filteredDoctors.length === 0 && (
                <tr>
                  <td colSpan={5} className="px-6 py-12 text-center text-gray-500">
                    No doctors found matching your search criteria.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* View Details Modal */}
      {isDetailsModalOpen && selectedDoctor && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-gray-900/50 backdrop-blur-sm">
          <motion.div 
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-white rounded-2xl shadow-xl w-full max-w-md overflow-hidden"
          >
            <div className="px-6 py-4 border-b border-gray-100 flex items-center justify-between bg-gray-50/50">
              <h2 className="text-lg font-semibold text-gray-900">Doctor Profile</h2>
              <button 
                onClick={() => setIsDetailsModalOpen(false)}
                className="text-gray-400 hover:text-gray-600 transition-colors"
              >
                <X size={20} />
              </button>
            </div>
            <div className="p-6">
              <div className="flex items-center gap-4 mb-6">
                <div className="w-16 h-16 rounded-full bg-[var(--color-primary)]/10 text-[var(--color-primary)] flex items-center justify-center font-medium text-xl">
                  {selectedDoctor.name.split(' ').map(n => n[0]).join('').substring(0, 2).toUpperCase()}
                </div>
                <div>
                  <h3 className="text-xl font-semibold text-gray-900">{selectedDoctor.name}</h3>
                  <p className="text-sm text-gray-500 font-mono">{selectedDoctor.id}</p>
                </div>
              </div>
              
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-gray-50 p-3 rounded-lg border border-gray-100">
                    <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-1">Status</p>
                    <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${
                      selectedDoctor.status === 'Active' 
                        ? 'bg-emerald-100 text-emerald-800' 
                        : 'bg-gray-200 text-gray-800'
                    }`}>
                      {selectedDoctor.status}
                    </span>
                  </div>
                  <div className="bg-gray-50 p-3 rounded-lg border border-gray-100">
                    <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-1">Joined</p>
                    <p className="text-sm text-gray-900 font-medium">{selectedDoctor.joinDate}</p>
                  </div>
                </div>

                <div className="bg-gray-50 p-4 rounded-lg border border-gray-100 space-y-3">
                  <div>
                    <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-1">Department & Specialty</p>
                    <p className="text-sm text-gray-900 font-medium">{selectedDoctor.department}</p>
                    <p className="text-sm text-gray-600">{selectedDoctor.specialty}</p>
                  </div>
                  <div className="pt-3 border-t border-gray-200">
                    <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Contact Information</p>
                    <div className="flex items-center gap-2 text-sm text-gray-700 mb-1.5">
                      <Mail size={14} className="text-gray-400" />
                      {selectedDoctor.email}
                    </div>
                    <div className="flex items-center gap-2 text-sm text-gray-700">
                      <Phone size={14} className="text-gray-400" />
                      {selectedDoctor.phone}
                    </div>
                  </div>
                </div>
              </div>

              <div className="mt-6 pt-6 border-t border-gray-100 flex justify-end gap-3">
                <button 
                  onClick={() => toggleDoctorStatus(selectedDoctor.id)}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    selectedDoctor.status === 'Active'
                      ? 'bg-red-50 text-red-600 hover:bg-red-100'
                      : 'bg-emerald-50 text-emerald-600 hover:bg-emerald-100'
                  }`}
                >
                  {selectedDoctor.status === 'Active' ? 'Deactivate Account' : 'Activate Account'}
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
              <button 
                onClick={() => setIsAddModalOpen(false)}
                className="text-gray-400 hover:text-gray-600 transition-colors"
              >
                <X size={20} />
              </button>
            </div>
            
            <form onSubmit={handleAddDoctor} className="p-6 space-y-4">
              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-gray-600 uppercase tracking-wider">Full Name</label>
                <input 
                  required
                  type="text" 
                  value={newDoctor.name}
                  onChange={e => setNewDoctor({...newDoctor, name: e.target.value})}
                  className="w-full bg-white border border-gray-300 rounded-lg px-3.5 py-2.5 focus:outline-none focus:border-[var(--color-primary)] focus:ring-1 focus:ring-[var(--color-primary)] text-sm shadow-sm" 
                  placeholder="Dr. Jane Doe" 
                />
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <label className="text-xs font-semibold text-gray-600 uppercase tracking-wider">Email</label>
                  <input 
                    required
                    type="email" 
                    value={newDoctor.email}
                    onChange={e => setNewDoctor({...newDoctor, email: e.target.value})}
                    className="w-full bg-white border border-gray-300 rounded-lg px-3.5 py-2.5 focus:outline-none focus:border-[var(--color-primary)] focus:ring-1 focus:ring-[var(--color-primary)] text-sm shadow-sm" 
                    placeholder="doctor@fetalai.com" 
                  />
                </div>
                <div className="space-y-1.5">
                  <label className="text-xs font-semibold text-gray-600 uppercase tracking-wider">Phone</label>
                  <input 
                    required
                    type="tel" 
                    value={newDoctor.phone}
                    onChange={e => setNewDoctor({...newDoctor, phone: e.target.value})}
                    className="w-full bg-white border border-gray-300 rounded-lg px-3.5 py-2.5 focus:outline-none focus:border-[var(--color-primary)] focus:ring-1 focus:ring-[var(--color-primary)] text-sm shadow-sm" 
                    placeholder="+1 (555) 000-0000" 
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <label className="text-xs font-semibold text-gray-600 uppercase tracking-wider">Department</label>
                  <select 
                    required
                    value={newDoctor.department}
                    onChange={e => setNewDoctor({...newDoctor, department: e.target.value})}
                    className="w-full bg-white border border-gray-300 rounded-lg px-3.5 py-2.5 focus:outline-none focus:border-[var(--color-primary)] focus:ring-1 focus:ring-[var(--color-primary)] text-sm shadow-sm"
                  >
                    <option value="">Select Department</option>
                    <option value="Obstetrics & Gynecology">Obstetrics & Gynecology</option>
                    <option value="Cardiology">Cardiology</option>
                    <option value="Imaging">Imaging</option>
                    <option value="Pediatrics">Pediatrics</option>
                  </select>
                </div>
                <div className="space-y-1.5">
                  <label className="text-xs font-semibold text-gray-600 uppercase tracking-wider">Specialty</label>
                  <input 
                    required
                    type="text" 
                    value={newDoctor.specialty}
                    onChange={e => setNewDoctor({...newDoctor, specialty: e.target.value})}
                    className="w-full bg-white border border-gray-300 rounded-lg px-3.5 py-2.5 focus:outline-none focus:border-[var(--color-primary)] focus:ring-1 focus:ring-[var(--color-primary)] text-sm shadow-sm" 
                    placeholder="e.g. Fetal Medicine Specialist" 
                  />
                </div>
              </div>

              <div className="pt-4 flex gap-3">
                <button 
                  type="button"
                  onClick={() => setIsAddModalOpen(false)}
                  className="flex-1 px-4 py-2.5 bg-white border border-gray-300 text-gray-700 rounded-lg text-sm font-medium hover:bg-gray-50 transition-colors shadow-sm"
                >
                  Cancel
                </button>
                <button 
                  type="submit"
                  className="flex-1 px-4 py-2.5 bg-[var(--color-primary)] text-white rounded-lg text-sm font-medium hover:bg-[var(--color-primary-light)] transition-colors shadow-sm"
                >
                  Create Account
                </button>
              </div>
            </form>
          </motion.div>
        </div>
      )}
    </div>
  );
}
