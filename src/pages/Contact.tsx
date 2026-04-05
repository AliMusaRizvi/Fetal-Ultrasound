import { motion } from 'motion/react';
import { Mail, Phone, MapPin } from 'lucide-react';

export default function Contact() {
  return (
    <div className="w-full bg-[var(--color-bg-warm)] min-h-[calc(100vh-80px)]">
      <section className="py-32 px-6 max-w-7xl mx-auto">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-20">
          {/* Contact Info */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="flex flex-col justify-center"
          >
            <span className="text-[var(--color-accent)] text-sm tracking-widest uppercase font-semibold mb-6 block">Get in Touch</span>
            <h1 className="text-5xl md:text-7xl font-serif text-[var(--color-primary)] mb-8 leading-tight">
              Let's connect <br />
              <span className="italic text-[var(--color-accent)]">for better care.</span>
            </h1>
            <p className="text-lg text-[var(--color-ink-light)] font-light leading-relaxed max-w-md mb-16">
              Whether you're a medical professional interested in our system or a researcher looking to collaborate, we'd love to hear from you.
            </p>

            <div className="space-y-8">
              {[
                { icon: <Mail size={24} />, title: "Email", info: "contact@fetalai.system" },
                { icon: <Phone size={24} />, title: "Phone", info: "+1 (555) 123-4567" },
                { icon: <MapPin size={24} />, title: "Research Center", info: "123 Medical Innovation Way, Suite 400" }
              ].map((item, idx) => (
                <div key={idx} className="flex items-start gap-6 group">
                  <div className="w-12 h-12 rounded-full bg-[var(--color-accent-light)] flex items-center justify-center text-[var(--color-primary)] group-hover:bg-[var(--color-primary)] group-hover:text-white transition-colors duration-300">
                    {item.icon}
                  </div>
                  <div>
                    <h4 className="font-serif text-xl text-[var(--color-primary)] mb-1">{item.title}</h4>
                    <p className="text-[var(--color-ink-light)] font-light">{item.info}</p>
                  </div>
                </div>
              ))}
            </div>
          </motion.div>

          {/* Contact Form */}
          <motion.div
            initial={{ opacity: 0, x: 30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="bg-white p-12 rounded-[40px] shadow-sm border border-[var(--color-ink)]/5 relative overflow-hidden"
          >
            <div className="absolute top-0 right-0 w-64 h-64 bg-[var(--color-accent-light)]/30 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2" />
            
            <h3 className="font-serif text-3xl text-[var(--color-primary)] mb-8 relative z-10">Send a Message</h3>
            
            <form className="space-y-6 relative z-10" onSubmit={(e) => e.preventDefault()}>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                <div className="space-y-2">
                  <label htmlFor="firstName" className="text-xs font-semibold tracking-widest uppercase text-[var(--color-ink-light)]">First Name</label>
                  <input
                    type="text"
                    id="firstName"
                    className="w-full bg-[var(--color-bg-warm)] border border-[var(--color-ink)]/10 rounded-xl px-4 py-3 focus:outline-none focus:border-[var(--color-primary)] focus:ring-1 focus:ring-[var(--color-primary)] transition-all"
                    placeholder="Jane"
                  />
                </div>
                <div className="space-y-2">
                  <label htmlFor="lastName" className="text-xs font-semibold tracking-widest uppercase text-[var(--color-ink-light)]">Last Name</label>
                  <input
                    type="text"
                    id="lastName"
                    className="w-full bg-[var(--color-bg-warm)] border border-[var(--color-ink)]/10 rounded-xl px-4 py-3 focus:outline-none focus:border-[var(--color-primary)] focus:ring-1 focus:ring-[var(--color-primary)] transition-all"
                    placeholder="Doe"
                  />
                </div>
              </div>
              
              <div className="space-y-2">
                <label htmlFor="email" className="text-xs font-semibold tracking-widest uppercase text-[var(--color-ink-light)]">Email Address</label>
                <input
                  type="email"
                  id="email"
                  className="w-full bg-[var(--color-bg-warm)] border border-[var(--color-ink)]/10 rounded-xl px-4 py-3 focus:outline-none focus:border-[var(--color-primary)] focus:ring-1 focus:ring-[var(--color-primary)] transition-all"
                  placeholder="jane.doe@hospital.org"
                />
              </div>

              <div className="space-y-2">
                <label htmlFor="message" className="text-xs font-semibold tracking-widest uppercase text-[var(--color-ink-light)]">Message</label>
                <textarea
                  id="message"
                  rows={4}
                  className="w-full bg-[var(--color-bg-warm)] border border-[var(--color-ink)]/10 rounded-xl px-4 py-3 focus:outline-none focus:border-[var(--color-primary)] focus:ring-1 focus:ring-[var(--color-primary)] transition-all resize-none"
                  placeholder="How can we help you?"
                />
              </div>

              <button
                type="submit"
                className="w-full bg-[var(--color-primary)] text-white font-medium tracking-wide py-4 rounded-xl hover:bg-[var(--color-primary-light)] transition-colors duration-300 mt-4"
              >
                Send Message
              </button>
            </form>
          </motion.div>
        </div>
      </section>
    </div>
  );
}
