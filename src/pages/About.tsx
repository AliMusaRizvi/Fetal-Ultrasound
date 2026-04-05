import { motion } from 'motion/react';

export default function About() {
  return (
    <div className="w-full bg-[var(--color-bg-warm)] min-h-screen">
      {/* Hero Section */}
      <section className="pt-32 pb-20 px-6 max-w-7xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="max-w-4xl"
        >
          <h1 className="text-5xl md:text-7xl font-serif text-[var(--color-primary)] mb-8 leading-tight">
            Our Mission to <br />
            <span className="italic text-[var(--color-accent)]">Transform Prenatal Care</span>
          </h1>
          <p className="text-xl text-[var(--color-ink-light)] font-light leading-relaxed max-w-2xl">
            Detecting fetal anomalies, especially in the brain, heart, and nuchal translucency (NT) regions, is critical for early medical intervention.
          </p>
        </motion.div>
      </section>

      {/* Image Grid */}
      <section className="py-12 px-6 max-w-7xl mx-auto">
        <div className="grid grid-cols-1 md:grid-cols-12 gap-6 h-[600px]">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
            className="md:col-span-8 rounded-[40px] overflow-hidden relative"
          >
            <img
              src="https://images.unsplash.com/photo-1579684385127-1ef15d508118?q=80&w=2080&auto=format&fit=crop"
              alt="Medical research"
              className="object-cover w-full h-full"
              referrerPolicy="no-referrer"
            />
            <div className="absolute inset-0 bg-[var(--color-primary)]/20 mix-blend-multiply" />
          </motion.div>
          
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="md:col-span-4 rounded-[40px] overflow-hidden relative bg-[var(--color-accent)]"
          >
            <img
              src="https://images.unsplash.com/photo-1530497610245-94d3c16cda28?q=80&w=1964&auto=format&fit=crop"
              alt="Technology in healthcare"
              className="object-cover w-full h-full opacity-60 mix-blend-luminosity"
              referrerPolicy="no-referrer"
            />
            <div className="absolute inset-0 p-8 flex flex-col justify-end text-white">
              <h3 className="font-serif text-3xl mb-4">The Need for AI</h3>
              <p className="font-light text-sm opacity-90">
                At present, this detection depends on manual image interpretation, which can be slow and inconsistent.
              </p>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Research Problem Statement */}
      <section className="py-32 px-6 max-w-7xl mx-auto">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-20 items-center">
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
          >
            <span className="text-[var(--color-accent)] text-sm tracking-widest uppercase font-semibold mb-6 block">Research Problem Statement</span>
            <h2 className="text-4xl md:text-5xl font-serif text-[var(--color-primary)] mb-8 leading-tight">
              A clear need for an <br />
              <span className="italic text-[var(--color-accent)]">AI-based system.</span>
            </h2>
            <div className="space-y-6 text-[var(--color-ink-light)] font-light leading-relaxed">
              <p>
                There is a clear need for an AI-based system that can automatically analyze ultrasound images and maternal data to assist doctors.
              </p>
              <p>
                Such a system can help identify possible anomalies, reduce human error, and generate reports that make diagnosis easier and more reliable. This can make diagnosis faster, more accurate, and more consistent — helping both mothers and babies get the care they need.
              </p>
            </div>
          </motion.div>
          
          <motion.div
            initial={{ opacity: 0, x: 30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="relative"
          >
            <div className="aspect-square rounded-full bg-[var(--color-primary-light)]/5 absolute -inset-10 -z-10 blur-3xl" />
            <div className="bg-white p-12 rounded-[40px] shadow-sm border border-[var(--color-ink)]/5">
              <h3 className="font-serif text-2xl text-[var(--color-primary)] mb-8">Key Focus Areas</h3>
              <ul className="space-y-6">
                {[
                  { title: "Brain Anomalies", desc: "Early detection of structural abnormalities in fetal brain development." },
                  { title: "Heart Defects", desc: "Identifying congenital heart conditions through automated image analysis." },
                  { title: "Nuchal Translucency (NT)", desc: "Precise measurement and analysis of the NT region for chromosomal risk assessment." }
                ].map((item, idx) => (
                  <li key={idx} className="flex gap-4">
                    <div className="w-8 h-8 rounded-full bg-[var(--color-accent-light)] text-[var(--color-primary)] flex items-center justify-center flex-shrink-0 font-serif font-bold text-sm">
                      {idx + 1}
                    </div>
                    <div>
                      <h4 className="font-medium text-[var(--color-primary)] mb-1">{item.title}</h4>
                      <p className="text-sm text-[var(--color-ink-light)] font-light">{item.desc}</p>
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          </motion.div>
        </div>
      </section>
    </div>
  );
}
