// src/components/AdminDashboard.jsx
import { useState } from 'react'
import { UploadIcon, TrainingIcon, ModelIcon } from './Icons'

export default function AdminDashboard() {
  const [activeTab, setActiveTab] = useState('upload')
  const [file, setFile] = useState(null)
  const [trainingStatus, setTrainingStatus] = useState(null)
  const [feedback, setFeedback] = useState(null)

  const handleFileUpload = async (e) => {
    e.preventDefault()
    if (!file) return
    
    const formData = new FormData()
    formData.append('file', file)
    
    try {
      const response = await fetch('http://localhost:5000/upload_data', {
        method: 'POST',
        body: formData
      })
      const data = await response.json()
      alert(`Upload successful: ${data.filename}`)
    } catch (error) {
      console.error('Upload error:', error)
      alert('Upload failed')
    }
  }

  const startTraining = async () => {
    setTrainingStatus('starting')
    try {
      const response = await fetch('http://localhost:5000/start_training', {
        method: 'POST'
      })
      const data = await response.json()
      console.log('data', data)
      setFeedback(data.error || '')
      setTrainingStatus('completed')
    } catch (error) {
      console.error('Training error:', error)
      setTrainingStatus('failed')
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-6xl mx-auto">
        <div className="flex justify-between items-center mb-8">
          <h1 className="text-3xl font-bold text-blue-800">Federated Learning Admin</h1>
          <div className="flex space-x-2">
            <span className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm">
              Model v2.1.4
            </span>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="flex border-b border-gray-200 mb-8">
          <button
            onClick={() => setActiveTab('upload')}
            className={`px-4 py-2 font-medium ${activeTab === 'upload' ? 'text-blue-600 border-b-2 border-blue-600' : 'text-gray-500'}`}
          >
            <UploadIcon className="inline mr-2" />
            Data Upload
          </button>
          <button
            onClick={() => setActiveTab('training')}
            className={`px-4 py-2 font-medium ${activeTab === 'training' ? 'text-blue-600 border-b-2 border-blue-600' : 'text-gray-500'}`}
          >
            <TrainingIcon className="inline mr-2" />
            Model Training
          </button>
          <button
            onClick={() => setActiveTab('performance')}
            className={`px-4 py-2 font-medium ${activeTab === 'performance' ? 'text-blue-600 border-b-2 border-blue-600' : 'text-gray-500'}`}
          >
            <ModelIcon className="inline mr-2" />
            Performance
          </button>
        </div>

        {/* Tab Content */}
        <div className="bg-white rounded-xl shadow-md p-6">
          {activeTab === 'upload' && (
            <div className="space-y-6">
              <h2 className="text-xl font-semibold">Upload Patient Data</h2>
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
                <div className="flex justify-center">
                  <UploadIcon className="h-12 w-12 text-gray-400" />
                </div>
                <div className="mt-4">
                  <label className="cursor-pointer bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-md">
                    Select CSV File
                    <input 
                      type="file" 
                      className="sr-only" 
                      accept=".csv"
                      onChange={(e) => setFile(e.target.files[0])}
                    />
                  </label>
                  <p className="mt-2 text-sm text-gray-600">
                    {file ? file.name : 'No file selected'}
                  </p>
                </div>
              </div>
              <div className="text-sm text-gray-500">
                <p>File requirements:</p>
                <ul className="list-disc pl-5 mt-2 space-y-1">
                  <li>CSV format with exact 16 columns</li>
                  <li>Column headers must match expected format</li>
                  <li>Max file size: 10MB</li>
                </ul>
              </div>
              <button
                onClick={handleFileUpload}
                disabled={!file}
                className={`w-full py-2 px-4 bg-blue-600 text-white rounded-md ${!file ? 'opacity-50 cursor-not-allowed' : 'hover:bg-blue-700'}`}
              >
                Upload Data
              </button>
            </div>
          )}

          {activeTab === 'training' && (
            <div className="space-y-6">
              <h2 className="text-xl font-semibold">Model Training</h2>
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <h3 className="font-medium text-blue-800">Federated Learning Process</h3>
                <p className="mt-2 text-sm text-blue-700">
                  Training occurs across multiple hospitals without sharing raw patient data.
                  Only model updates are aggregated.
                </p>
              </div>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Training Rounds
                  </label>
                  <select className="w-full border border-gray-300 rounded-md p-2">
                    <option>3 rounds (default)</option>
                    <option>5 rounds</option>
                    <option>10 rounds</option>
                  </select>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Client Participation
                  </label>
                  <input 
                    type="range" 
                    min="1" 
                    max="10" 
                    defaultValue="3"
                    className="w-full"
                  />
                  <div className="flex justify-between text-xs text-gray-500 mt-1">
                    <span>1 hospital</span>
                    <span>10 hospitals</span>
                  </div>
                </div>
              </div>
              
              <button
                onClick={startTraining}
                disabled={trainingStatus === 'starting'}
                className={`w-full py-2 px-4 bg-green-600 text-white rounded-md flex items-center justify-center ${trainingStatus === 'starting' ? 'opacity-50 cursor-not-allowed' : 'hover:bg-green-700'}`}
              >
                {trainingStatus === 'starting' ? (
                  <>
                    <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Starting Training...
                  </>
                ) : 'Start Federated Training'}
              </button>
              
              {trainingStatus === 'completed' && (
                <div className={`${feedback ? "bg-red-50 border-red-200 text-red-800" : "bg-green-50 border-green-200 text-green-800"} border rounded-lg p-4`}>
                  {feedback ? feedback : 'Training completed successfully! Model has been updated.'}
                </div>
              )}
            </div>
          )}

          {activeTab === 'performance' && (
            <div className="space-y-6">
              <h2 className="text-xl font-semibold">Model Performance</h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-white border border-gray-200 rounded-lg p-4 shadow-sm">
                  <h3 className="text-sm font-medium text-gray-500">Accuracy</h3>
                  <p className="mt-1 text-2xl font-semibold text-blue-600">87.2%</p>
                  <p className="text-xs text-gray-500 mt-1">+2.1% from last version</p>
                </div>
                <div className="bg-white border border-gray-200 rounded-lg p-4 shadow-sm">
                  <h3 className="text-sm font-medium text-gray-500">Precision</h3>
                  <p className="mt-1 text-2xl font-semibold text-green-600">89.5%</p>
                  <p className="text-xs text-gray-500 mt-1">+1.8% from last version</p>
                </div>
                <div className="bg-white border border-gray-200 rounded-lg p-4 shadow-sm">
                  <h3 className="text-sm font-medium text-gray-500">Recall</h3>
                  <p className="mt-1 text-2xl font-semibold text-purple-600">85.7%</p>
                  <p className="text-xs text-gray-500 mt-1">+3.2% from last version</p>
                </div>
              </div>
              
              <div className="bg-white border border-gray-200 rounded-lg p-4 shadow-sm">
                <h3 className="font-medium mb-2">Accuracy Over Time</h3>
                <div className="h-64 bg-gray-50 rounded flex items-center justify-center text-gray-400">
                  [Accuracy chart visualization]
                </div>
              </div>
              
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <h3 className="font-medium text-blue-800">Active Hospitals</h3>
                <div className="mt-2 space-y-2">
                  {['General Hospital', 'City Medical', 'University Clinic'].map(hospital => (
                    <div key={hospital} className="flex items-center">
                      <div className="h-2 w-2 bg-green-500 rounded-full mr-2"></div>
                      <span className="text-sm">{hospital}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}