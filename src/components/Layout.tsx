import { Outlet, Link, useLocation } from 'react-router-dom';
import { motion } from 'motion/react';
import { Activity } from 'lucide-react';

export default function Layout() {
  const location = useLocation();

  return (
    <div className="min-h-screen flex flex-col bg-[var(--color-bg-warm)] selection:bg-[var(--color-accent-light)] selection:text-[var(--color-primary)]">
      {/* Navigation */}
      <header className="sticky top-0 z-50 glass-panel border-b border-white/20">
        <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
          <Link to="/" className="flex items-center gap-3 group">
            <div className="w-10 h-10 rounded-full bg-[var(--color-primary)] flex items-center justify-center text-white group-hover:bg-[var(--color-accent)] transition-colors duration-300">
              <Activity size={20} strokeWidth={1.5} />
            </div>
            <span className="font-serif text-xl font-medium tracking-wide text-[var(--color-primary)]">
              Fetal<span className="italic text-[var(--color-accent)]">AI</span>
            </span>
          </Link>

          <nav className="hidden md:flex items-center gap-8">
            {[
              { path: '/', label: 'Home' },
              { path: '/about', label: 'About Us' },
              { path: '/contact', label: 'Contact' },
            ].map((link) => (
              <Link
                key={link.path}
                to={link.path}
                className={`text-sm tracking-widest uppercase transition-all duration-300 relative ${
                  location.pathname === link.path
                    ? 'text-[var(--color-primary)] font-medium'
                    : 'text-[var(--color-ink-light)] hover:text-[var(--color-primary)]'
                }`}
              >
                {link.label}
                {location.pathname === link.path && (
                  <motion.div
                    layoutId="nav-indicator"
                    className="absolute -bottom-2 left-0 right-0 h-[1px] bg-[var(--color-primary)]"
                    initial={false}
                    transition={{ type: 'spring', stiffness: 300, damping: 30 }}
                  />
                )}
              </Link>
            ))}
          </nav>

          <div className="flex items-center gap-4">
            <Link
              to="/login"
              className="px-6 py-2.5 rounded-full border border-[var(--color-primary)] text-[var(--color-primary)] text-sm tracking-widest uppercase hover:bg-[var(--color-primary)] hover:text-white transition-all duration-300"
            >
              Doctor Portal
            </Link>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-grow">
        <Outlet />
      </main>

      {/* Footer */}
      <footer className="bg-[var(--color-primary)] text-white/80 py-16 border-t border-white/10">
        <div className="max-w-7xl mx-auto px-6 grid grid-cols-1 md:grid-cols-4 gap-12">
          <div className="col-span-1 md:col-span-2">
            <Link to="/" className="flex items-center gap-3 mb-6">
              <div className="w-10 h-10 rounded-full bg-white/10 flex items-center justify-center text-white">
                <Activity size={20} strokeWidth={1.5} />
              </div>
              <span className="font-serif text-2xl font-medium tracking-wide text-white">
                Fetal<span className="italic text-[var(--color-accent-light)]">AI</span>
              </span>
            </Link>
            <p className="text-sm leading-relaxed max-w-sm font-sans font-light">
              Empowering medical professionals with AI-assisted fetal anomaly detection. 
              Reducing errors, saving time, and making prenatal care more reliable.
            </p>
          </div>
          
          <div>
            <h4 className="font-serif text-lg mb-6 text-white">Navigation</h4>
            <ul className="space-y-4 text-sm font-light">
              <li><Link to="/" className="hover:text-white transition-colors">Home</Link></li>
              <li><Link to="/about" className="hover:text-white transition-colors">About Us</Link></li>
              <li><Link to="/contact" className="hover:text-white transition-colors">Contact</Link></li>
            </ul>
          </div>

          <div>
            <h4 className="font-serif text-lg mb-6 text-white">System</h4>
            <ul className="space-y-4 text-sm font-light">
              <li><Link to="/login" className="hover:text-white transition-colors">Doctor Login</Link></li>
              <li><a href="#" className="hover:text-white transition-colors">Privacy Policy</a></li>
              <li><a href="#" className="hover:text-white transition-colors">Terms of Service</a></li>
            </ul>
          </div>
        </div>
        <div className="max-w-7xl mx-auto px-6 mt-16 pt-8 border-t border-white/10 flex flex-col md:flex-row justify-between items-center text-xs font-light">
          <p>&copy; {new Date().getFullYear()} FetalAI Detection System. All rights reserved.</p>
          <p className="mt-2 md:mt-0">For medical professional use only.</p>
        </div>
      </footer>
    </div>
  );
}
