// src/components/RiskMeter.jsx
export default function RiskMeter({ prediction }) {
    const riskLevels = [
      { level: 'Very High', threshold: 0.8, color: 'bg-red-600' },
      { level: 'High', threshold: 0.7, color: 'bg-orange-500' },
      { level: 'Moderate', threshold: 0.6, color: 'bg-yellow-400' },
      { level: 'Borderline', threshold: 0.55, color: 'bg-blue-400' },
      { level: 'Uncertain', threshold: 0.45, color: 'bg-gray-300' },
      { level: 'Low', threshold: 0, color: 'bg-green-400' }
    ]
  
    const currentLevel = riskLevels.find(
      level => prediction.prediction >= level.threshold
    )
  
    return (
      <div className="bg-white rounded-xl shadow-md p-6">
        <div className="flex justify-between items-center mb-2">
          <h2 className="text-xl font-semibold text-gray-800">Risk Assessment</h2>
          <span className={`px-3 py-1 rounded-full text-xs font-medium ${currentLevel.color.replace('bg', 'text').replace('400', '800').replace('300', '600')} ${currentLevel.color.replace('400', '100').replace('300', '100')}`}>
            {prediction.risk_level}
          </span>
        </div>
        
        <div className="mt-4">
          <div className="flex justify-between text-sm text-gray-600 mb-1">
            <span>0%</span>
            <span>100%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-4">
            <div
              className={`h-4 rounded-full ${currentLevel.color}`}
              style={{ width: `${prediction.prediction * 100}%` }}
            ></div>
          </div>
        </div>
        
        <div className="mt-6">
          <p className="text-gray-700">
            <span className="font-medium">Prediction Score:</span> {(prediction.prediction * 100).toFixed(1)}%
          </p>
          <p className="text-gray-700">
            <span className="font-medium">Confidence:</span> {prediction.confidence}
          </p>
          <p className="mt-2 text-gray-600">{prediction.interpretation}</p>
        </div>
      </div>
    )
  }