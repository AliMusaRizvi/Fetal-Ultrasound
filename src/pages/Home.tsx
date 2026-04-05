import { motion } from 'motion/react';
import { ArrowRight, BrainCircuit, HeartPulse, FileText, ShieldCheck } from 'lucide-react';
import { Link } from 'react-router-dom';

export default function Home() {
  return (
    <div className="w-full">
      {/* Hero Section */}
      <section className="relative min-h-[90vh] flex items-center justify-center overflow-hidden bg-[var(--color-bg-warm)]">
        {/* Abstract Background Elements */}
        <div className="absolute top-0 right-0 w-[800px] h-[800px] bg-[var(--color-accent-light)] rounded-full blur-[120px] opacity-40 -translate-y-1/2 translate-x-1/3" />
        <div className="absolute bottom-0 left-0 w-[600px] h-[600px] bg-[var(--color-primary-light)] rounded-full blur-[100px] opacity-10 translate-y-1/3 -translate-x-1/4" />

        <div className="max-w-7xl mx-auto px-6 grid grid-cols-1 lg:grid-cols-2 gap-16 items-center relative z-10 py-20">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, ease: "easeOut" }}
            className="flex flex-col items-start"
          >
            <h1 className="text-6xl md:text-7xl lg:text-[80px] leading-[0.9] font-serif text-[var(--color-primary)] mb-8 tracking-tight">
              Precision in <br />
              <span className="text-[var(--color-accent)] italic">Every Scan.</span>
            </h1>
            
            <p className="text-lg md:text-xl text-[var(--color-ink-light)] font-light leading-relaxed max-w-xl mb-12">
              Fetal anomaly detection is one of the most important steps in ensuring a healthy pregnancy. 
              Our AI system assists doctors in spotting potential issues early by combining ultrasound images with clinical data.
            </p>
            
            <div className="flex flex-col sm:flex-row items-center gap-6">
              <Link
                to="/login"
                className="px-8 py-4 rounded-full bg-[var(--color-primary)] text-white font-medium tracking-wide hover:bg-[var(--color-primary-light)] transition-all duration-300 flex items-center gap-3 group w-full sm:w-auto justify-center"
              >
                Doctor Login
                <ArrowRight size={18} className="group-hover:translate-x-1 transition-transform" />
              </Link>
              <Link
                to="/about"
                className="px-8 py-4 rounded-full border border-[var(--color-ink)]/20 text-[var(--color-ink)] font-medium tracking-wide hover:bg-[var(--color-ink)]/5 transition-all duration-300 w-full sm:w-auto text-center"
              >
                Learn More
              </Link>
            </div>
          </motion.div>

          {/* Hero Image / Graphic */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 1, delay: 0.2, ease: "easeOut" }}
            className="relative h-[600px] w-full flex items-center justify-center"
          >
            <div className="absolute inset-0 bg-gradient-to-tr from-[var(--color-primary-light)] to-[var(--color-accent)] rounded-[40px] opacity-10 rotate-3 scale-105" />
            <img
              src="https://images.unsplash.com/photo-1584515933487-779824d29309?q=80&w=2070&auto=format&fit=crop"
              alt="Medical professional reviewing scans"
              className="object-cover w-full h-full rounded-[40px] shadow-2xl"
              referrerPolicy="no-referrer"
            />
            {/* Floating UI Element */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 1, duration: 0.8 }}
              className="absolute -bottom-8 -left-8 glass-panel p-6 rounded-2xl shadow-xl flex items-center gap-4 border border-white/40"
            >
              <div className="w-12 h-12 rounded-full bg-[var(--color-primary)]/10 flex items-center justify-center text-[var(--color-primary)]">
                <ShieldCheck size={24} />
              </div>
              <div>
                <p className="text-sm font-bold text-[var(--color-primary)]">Analysis Complete</p>
                <p className="text-xs text-[var(--color-ink-light)]">No anomalies detected</p>
              </div>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* Problem Domain Section */}
      <section className="py-32 bg-white relative">
        <div className="max-w-7xl mx-auto px-6">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-16 items-start">
            <div className="lg:col-span-5 lg:sticky lg:top-32">
              <h2 className="text-4xl md:text-5xl font-serif text-[var(--color-primary)] mb-6 leading-tight">
                The Challenge of <br />
                <span className="italic text-[var(--color-accent)]">Early Detection</span>
              </h2>
              <div className="w-20 h-[1px] bg-[var(--color-accent)] mb-8" />
              <p className="text-[var(--color-ink-light)] font-light leading-relaxed mb-8">
                Early detection of fetal anomalies is one of the most important parts of prenatal care. Doctors usually rely on radiologists and sonographers to manually examine ultrasound images and identify potential issues.
              </p>
              <p className="text-[var(--color-ink-light)] font-light leading-relaxed">
                However, this process takes time, depends on human expertise, and may lead to variation in diagnosis. In many cases, especially in developing regions, limited access to specialists makes early detection difficult.
              </p>
            </div>

            <div className="lg:col-span-7 grid grid-cols-1 sm:grid-cols-2 gap-8">
              {[
                {
                  icon: <BrainCircuit size={32} />,
                  title: "Subjective Interpretation",
                  desc: "Manual examination depends heavily on human expertise, leading to potential variation in diagnosis."
                },
                {
                  icon: <HeartPulse size={32} />,
                  title: "Time-Intensive",
                  desc: "Analyzing complex ultrasound images takes significant time, delaying critical medical interventions."
                },
                {
                  icon: <FileText size={32} />,
                  title: "Limited Accessibility",
                  desc: "Developing regions often lack access to specialized radiologists, making early detection difficult."
                }
              ].map((item, idx) => (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true, margin: "-100px" }}
                  transition={{ duration: 0.6, delay: idx * 0.1 }}
                  className={`p-8 rounded-3xl bg-[var(--color-bg-warm)] border border-[var(--color-ink)]/5 hover:border-[var(--color-accent)]/30 transition-colors duration-500 ${idx === 2 ? 'sm:col-span-2 sm:w-1/2 sm:mx-auto' : ''}`}
                >
                  <div className="w-14 h-14 rounded-2xl bg-white shadow-sm flex items-center justify-center text-[var(--color-accent)] mb-6">
                    {item.icon}
                  </div>
                  <h3 className="text-xl font-serif font-medium text-[var(--color-primary)] mb-4">{item.title}</h3>
                  <p className="text-sm text-[var(--color-ink-light)] font-light leading-relaxed">{item.desc}</p>
                </motion.div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Solution Section */}
      <section className="py-32 bg-[var(--color-primary)] text-white relative overflow-hidden">
        <div className="absolute inset-0 opacity-10" style={{ backgroundImage: 'radial-gradient(circle at 2px 2px, white 1px, transparent 0)', backgroundSize: '40px 40px' }} />
        
        <div className="max-w-7xl mx-auto px-6 relative z-10 text-center">
          <span className="text-[var(--color-accent-light)] text-sm tracking-widest uppercase font-semibold mb-4 block">Our Approach</span>
          <h2 className="text-4xl md:text-6xl font-serif mb-8 max-w-4xl mx-auto leading-tight">
            Supporting doctors, <br />
            <span className="italic text-[var(--color-accent-light)]">not replacing them.</span>
          </h2>
          <p className="text-lg text-white/70 font-light max-w-2xl mx-auto mb-20 leading-relaxed">
            With the rise of Artificial Intelligence, there is now a chance to support doctors through automated systems that can analyze ultrasound images and clinical data together.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-12 text-left">
            {[
              {
                num: "01",
                title: "Data Integration",
                desc: "Combining ultrasound images with the mother's clinical data for a comprehensive view."
              },
              {
                num: "02",
                title: "AI Analysis",
                desc: "Automated analysis to identify possible anomalies in the brain, heart, and NT regions."
              },
              {
                num: "03",
                title: "Clear Reporting",
                desc: "Generates an easy-to-understand report highlighting detected anomalies and next steps."
              }
            ].map((step, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: idx * 0.2 }}
                className="relative"
              >
                <div className="text-6xl font-serif text-white/10 font-bold mb-6">{step.num}</div>
                <h3 className="text-2xl font-serif mb-4">{step.title}</h3>
                <p className="text-white/60 font-light leading-relaxed">{step.desc}</p>
                {idx < 2 && (
                  <div className="hidden md:block absolute top-12 -right-6 w-12 h-[1px] bg-white/20" />
                )}
              </motion.div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}
