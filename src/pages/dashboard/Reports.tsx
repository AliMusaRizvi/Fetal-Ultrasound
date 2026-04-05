export default function Reports() {
  return (
    <div className="max-w-6xl mx-auto space-y-6 sm:space-y-8">
      <div className="mb-6 sm:mb-8">
        <h1 className="text-xl sm:text-2xl font-semibold text-gray-900 tracking-tight mb-2">Diagnostic Reports</h1>
        <p className="text-gray-500 text-sm">
          View and manage all patient diagnostic reports and AI analysis results.
        </p>
      </div>

      <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden flex flex-col">
        <div className="p-4 sm:p-5 border-b border-gray-100 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 bg-gray-50/50">
          <div className="flex flex-col sm:flex-row gap-3 sm:gap-4 w-full sm:w-auto">
            <input 
              type="text" 
              placeholder="Search reports..." 
              className="bg-white border border-gray-300 rounded-lg px-3.5 py-2 text-sm focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 shadow-sm w-full sm:w-64"
            />
            <select className="bg-white border border-gray-300 rounded-lg px-3.5 py-2 text-sm focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 shadow-sm text-gray-700 w-full sm:w-auto">
              <option>All Statuses</option>
              <option>Normal</option>
              <option>Review Required</option>
            </select>
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse min-w-[600px]">
            <thead>
              <tr className="bg-white text-xs uppercase tracking-wider text-gray-500 border-b border-gray-100">
                <th className="px-4 sm:px-6 py-4 font-medium">Report ID</th>
                <th className="px-4 sm:px-6 py-4 font-medium">Patient ID</th>
                <th className="px-4 sm:px-6 py-4 font-medium">Date</th>
                <th className="px-4 sm:px-6 py-4 font-medium">Findings</th>
                <th className="px-4 sm:px-6 py-4 font-medium text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="text-sm divide-y divide-gray-50">
              {[
                { id: 'REP-1029', pt: 'PT-8921', date: 'Oct 24, 2023', findings: 'Normal', color: 'text-emerald-700 bg-emerald-50 border-emerald-200' },
                { id: 'REP-1028', pt: 'PT-8922', date: 'Oct 24, 2023', findings: 'VSD Detected', color: 'text-amber-700 bg-amber-50 border-amber-200' },
                { id: 'REP-1027', pt: 'PT-8923', date: 'Oct 23, 2023', findings: 'Normal', color: 'text-emerald-700 bg-emerald-50 border-emerald-200' },
                { id: 'REP-1026', pt: 'PT-8910', date: 'Oct 22, 2023', findings: 'NT Enlarged', color: 'text-red-700 bg-red-50 border-red-200' },
              ].map((row, idx) => (
                <tr key={idx} className="hover:bg-gray-50/50 transition-colors">
                  <td className="px-4 sm:px-6 py-4 font-mono text-gray-500">{row.id}</td>
                  <td className="px-4 sm:px-6 py-4 font-medium text-gray-900">{row.pt}</td>
                  <td className="px-4 sm:px-6 py-4 text-gray-500 whitespace-nowrap">{row.date}</td>
                  <td className="px-4 sm:px-6 py-4">
                    <span className={`px-2.5 py-1 rounded-md text-xs font-medium border whitespace-nowrap ${row.color}`}>
                      {row.findings}
                    </span>
                  </td>
                  <td className="px-4 sm:px-6 py-4 text-right whitespace-nowrap">
                    <button className="text-blue-600 hover:text-blue-700 font-medium transition-colors mr-3 sm:mr-4">View</button>
                    <button className="text-gray-600 hover:text-gray-900 font-medium transition-colors">Download</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
