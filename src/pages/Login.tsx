import React, { useState } from 'react';
import { motion } from 'motion/react';
import { Lock, Mail, ArrowRight, Fingerprint, Activity } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { signIn } from '../lib/auth';

export default function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      const user = await signIn(email, password);
      localStorage.setItem('userRole', user.role);
      localStorage.setItem('userName', user.full_name);
      localStorage.setItem('userId', user.id);
      navigate('/dashboard');
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Login failed. Please check your credentials.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex bg-[var(--color-bg-warm)]">
      {/* Left Pane - Login Form */}
      <div className="w-full lg:w-1/2 flex flex-col relative z-10">
        <div className="p-8">
          <div className="flex items-center gap-2">
            <div className="w-10 h-10 rounded-xl bg-[var(--color-primary)] text-white flex items-center justify-center">
              <Activity size={24} />
            </div>
            <span className="text-xl font-semibold tracking-tight text-[var(--color-primary)]">Fetal<span className="font-light italic text-[var(--color-accent)]">AI</span></span>
          </div>
        </div>
        <div className="flex-grow flex flex-col justify-center px-8 md:px-16 lg:px-24 max-w-2xl mx-auto w-full">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, ease: "easeOut" }}
          >
            <div className="mb-12">
              <h1 className="text-4xl md:text-5xl font-serif text-[var(--color-primary)] mb-4 leading-tight">
                Welcome <span className="italic text-[var(--color-accent)]">Back</span>
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
                <label htmlFor="email" className="text-xs font-semibold tracking-widest uppercase text-[var(--color-ink-light)] ml-1">Email Address</label>
                <div className="relative group">
                  <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none text-[var(--color-ink-light)] group-focus-within:text-[var(--color-primary)] transition-colors">
                    <Mail size={20} strokeWidth={1.5} />
                  </div>
                  <input
                    type="email"
                    id="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    required
                    className="w-full bg-white border border-[var(--color-ink)]/10 rounded-2xl pl-12 pr-4 py-4 focus:outline-none focus:border-[var(--color-primary)] focus:ring-1 focus:ring-[var(--color-primary)] transition-all font-mono text-sm shadow-sm"
                    placeholder="doctor@fetalai.com"
                  />
                </div>
              </div>

              <div className="space-y-2">
                <div className="flex justify-between items-center ml-1">
                  <label htmlFor="password" className="text-xs font-semibold tracking-widest uppercase text-[var(--color-ink-light)]">Password</label>
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
                    required
                    className="w-full bg-white border border-[var(--color-ink)]/10 rounded-2xl pl-12 pr-4 py-4 focus:outline-none focus:border-[var(--color-primary)] focus:ring-1 focus:ring-[var(--color-primary)] transition-all font-mono text-sm tracking-widest shadow-sm"
                    placeholder="••••••••"
                  />
                </div>
              </div>

              <button
                type="submit"
                disabled={loading}
                className="w-full bg-[var(--color-primary)] text-white font-medium tracking-wide py-4 rounded-2xl hover:bg-[var(--color-primary-light)] transition-all duration-300 mt-8 flex items-center justify-center gap-3 group shadow-xl shadow-[var(--color-primary)]/20 disabled:opacity-60 disabled:cursor-not-allowed"
              >
                {loading ? (
                  <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                ) : (
                  <>
                    <Fingerprint size={20} className="opacity-70 group-hover:opacity-100 transition-opacity" />
                    Authenticate
                    <ArrowRight size={18} className="group-hover:translate-x-1 transition-transform" />
                  </>
                )}
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

      {/* Right Pane */}
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
