// src/components/FeatureCard.jsx
export default function FeatureCard({ feature }) {
    const directionColors = {
      Positive: 'text-red-500',
      Negative: 'text-green-500'
    }
  
    return (
      <div className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
        <div className="flex justify-between items-start">
          <h3 className="font-medium text-gray-800 capitalize">
            {feature.feature.replace(/([A-Z])/g, ' $1').trim()}
          </h3>
          <span className={`text-xs font-semibold ${directionColors[feature.direction]}`}>
            {feature.direction}
          </span>
        </div>
        
        <div className="mt-2">
          <div className="flex justify-between text-xs text-gray-500 mb-1">
            <span>Influence</span>
            <span>{(feature.importance * 100).toFixed(1)}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className="h-2 rounded-full bg-blue-500"
              style={{ width: `${Math.min(feature.importance * 1000, 100)}%` }}
            ></div>
          </div>
        </div>
        
        <div className="mt-3">
          <p className="text-sm text-gray-600">{feature.clinical_note}</p>
          {feature.recommended_action && (
            <p className="mt-1 text-xs font-medium text-blue-600">
              {feature.recommended_action}
            </p>
          )}
        </div>
      </div>
    )
  }