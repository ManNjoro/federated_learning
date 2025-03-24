// src/App.jsx
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import PredictionDashboard from './components/PredictionDashboard'
import AdminDashboard from './components/AdminDashboard'

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<PredictionDashboard />} />
        <Route path="/admin" element={
            <AdminDashboard />
        } />
      </Routes>
    </Router>
  )
}

export default App