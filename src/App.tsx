import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Home from './pages/Home';
import About from './pages/About';
import Contact from './pages/Contact';
import Login from './pages/Login';
import DashboardLayout from './components/DashboardLayout';
import DashboardHome from './pages/dashboard/DashboardHome';
import UploadScan from './pages/dashboard/UploadScan';
import Reports from './pages/dashboard/Reports';
import ManageDoctors from './pages/dashboard/ManageDoctors';
import AnalysisHistory from './pages/dashboard/AnalysisHistory';
import Alerts from './pages/dashboard/Alerts';

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Home />} />
          <Route path="about" element={<About />} />
          <Route path="contact" element={<Contact />} />
        </Route>
        
        <Route path="/login" element={<Login />} />

        <Route path="/dashboard" element={<DashboardLayout />}>
          <Route index element={<DashboardHome />} />
          <Route path="analyze" element={<UploadScan />} />
          <Route path="reports" element={<Reports />} />
          <Route path="history" element={<AnalysisHistory />} />
          <Route path="doctors" element={<ManageDoctors />} />
          <Route path="alerts" element={<Alerts />} />
        </Route>
      </Routes>
    </Router>
  );
}
