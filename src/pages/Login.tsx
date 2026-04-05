import React, { useState } from 'react';
import { motion } from 'motion/react';
import { Lock, User, ArrowRight, Fingerprint } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

export default function Login() {
  const [doctorId, setDoctorId] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault();
    if (doctorId === 'admin' && password === 'admin') {
      localStorage.setItem('userRole', 'admin');
      localStorage.setItem('userName', 'System Admin');
      localStorage.setItem('userId', 'ADMIN');
      navigate('/dashboard');
    } else if (doctorId === '0000' && password === 'admin') {
      localStorage.setItem('userRole', 'doctor');
      localStorage.setItem('userName', 'Dr. Sarah Jenkins');
      localStorage.setItem('userId', 'DOC-0000');
      navigate('/dashboard');
    } else {
      setError('Invalid credentials. Try 0000 / admin or admin / admin');
    }
  };

  return (
    <div className="min-h-screen flex bg-[var(--color-bg-warm)]">
      {/* Left Pane - Login Form */}
      <div className="w-full lg:w-1/2 flex flex-col relative z-10">
        <div className="flex-grow flex flex-col justify-center px-8 md:px-16 lg:px-24 max-w-2xl mx-auto w-full">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, ease: "easeOut" }}
          >
            <div className="mb-12">
              <h1 className="text-4xl md:text-5xl font-serif text-[var(--color-primary)] mb-4 leading-tight">
                Doctor <span className="italic text-[var(--color-accent)]">Portal</span>
              </h1>
              <p className="text-[var(--color-ink-light)] font-light text-lg">
                Secure access to the Fetal Anomaly Detection System.
              </p>
            </div>

            <form className="space-y-6" onSubmit={handleLogin}>
              {error && (
                <div className="p-4 rounded-xl bg-red-50 border border-red-100 text-red-600 text-sm font-medium">
                  {error}
                </div>
              )}
              <div className="space-y-2">
                <label htmlFor="doctorId" className="text-xs font-semibold tracking-widest uppercase text-[var(--color-ink-light)] ml-1">Medical ID / Admin ID</label>
                <div className="relative group">
                  <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none text-[var(--color-ink-light)] group-focus-within:text-[var(--color-primary)] transition-colors">
                    <User size={20} strokeWidth={1.5} />
                  </div>
                  <input
                    type="text"
                    id="doctorId"
                    value={doctorId}
                    onChange={(e) => setDoctorId(e.target.value)}
                    className="w-full bg-white border border-[var(--color-ink)]/10 rounded-2xl pl-12 pr-4 py-4 focus:outline-none focus:border-[var(--color-primary)] focus:ring-1 focus:ring-[var(--color-primary)] transition-all font-mono text-sm shadow-sm"
                    placeholder="DOC-XXXXX or admin"
                  />
                </div>
              </div>

              <div className="space-y-2">
                <div className="flex justify-between items-center ml-1">
                  <label htmlFor="password" className="text-xs font-semibold tracking-widest uppercase text-[var(--color-ink-light)]">Password</label>
                  <a href="#" className="text-xs text-[var(--color-accent)] hover:text-[var(--color-primary)] transition-colors font-medium">Forgot Password?</a>
                </div>
                <div className="relative group">
                  <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none text-[var(--color-ink-light)] group-focus-within:text-[var(--color-primary)] transition-colors">
                    <Lock size={20} strokeWidth={1.5} />
                  </div>
                  <input
                    type="password"
                    id="password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className="w-full bg-white border border-[var(--color-ink)]/10 rounded-2xl pl-12 pr-4 py-4 focus:outline-none focus:border-[var(--color-primary)] focus:ring-1 focus:ring-[var(--color-primary)] transition-all font-mono text-sm tracking-widest shadow-sm"
                    placeholder="••••••••"
                  />
                </div>
              </div>

              <button
                type="submit"
                className="w-full bg-[var(--color-primary)] text-white font-medium tracking-wide py-4 rounded-2xl hover:bg-[var(--color-primary-light)] transition-all duration-300 mt-8 flex items-center justify-center gap-3 group shadow-xl shadow-[var(--color-primary)]/20"
              >
                <Fingerprint size={20} className="opacity-70 group-hover:opacity-100 transition-opacity" />
                Authenticate
                <ArrowRight size={18} className="group-hover:translate-x-1 transition-transform" />
              </button>
            </form>
          </motion.div>
        </div>
        
        <div className="p-8 md:p-12 flex-shrink-0 text-center lg:text-left">
          <p className="text-xs text-[var(--color-ink-light)] font-light">
            &copy; {new Date().getFullYear()} FetalAI Detection System. <br className="lg:hidden" />
            <span className="hidden lg:inline"> | </span> 
            Protected by advanced encryption.
          </p>
        </div>
      </div>

      {/* Right Pane - Image & Atmosphere */}
      <div className="hidden lg:block lg:w-1/2 relative overflow-hidden bg-[var(--color-primary)]">
        <div className="absolute inset-0 bg-[var(--color-primary)] mix-blend-multiply z-10 opacity-60" />
        <div className="absolute inset-0 bg-gradient-to-br from-[var(--color-primary)]/80 via-transparent to-[var(--color-accent)]/40 z-10" />
        
        <motion.img
          initial={{ scale: 1.05 }}
          animate={{ scale: 1 }}
          transition={{ duration: 1.5, ease: "easeOut" }}
          src="https://images.unsplash.com/photo-1559839734-2b71ea197ec2?q=80&w=2070&auto=format&fit=crop"
          alt="Medical technology background"
          className="absolute inset-0 w-full h-full object-cover"
          referrerPolicy="no-referrer"
        />

        {/* Floating Elements */}
        <div className="absolute inset-0 z-20 p-16 flex flex-col justify-end">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3, duration: 0.8 }}
            className="max-w-md"
          >
            <div className="w-12 h-[1px] bg-[var(--color-accent-light)] mb-8" />
            <h2 className="text-4xl font-serif text-white mb-6 leading-tight">
              Precision in <br />
              <span className="italic text-[var(--color-accent-light)]">Every Scan.</span>
            </h2>
            <p className="text-white/70 font-light leading-relaxed">
              Combining ultrasound imaging with clinical data to support early detection of fetal anomalies. Empowering doctors with AI-driven insights.
            </p>
          </motion.div>
        </div>
      </div>
    </div>
  );
}
