// src/components/PredictionDashboard.jsx
import { useState } from "react";
import RiskMeter from "./RiskMeter";
import FeatureCard from "./FeatureCard";

export default function PredictionDashboard() {
  const [formData, setFormData] = useState({
    Age: "",
    Gender: "Male",
    Polyuria: "No",
    Polydipsia: "No",
    "sudden weight loss": "No",
    weakness: "No",
    Polyphagia: "No",
    "Genital thrush": "No",
    "visual blurring": "No",
    Itching: "No",
    Irritability: "No",
    "delayed healing": "No",
    "partial paresis": "No",
    "muscle stiffness": "No",
    Alopecia: "No",
    Obesity: "No",

    // ... other fields
  });
  // const buildDict = {};
  const symptoms = [
    "Polyuria",
    "Polydipsia",
    "sudden weight loss",
    "weakness",
    "Polyphagia",
    "Genital thrush",
    "visual blurring",
    "Itching",
    "Irritability",
    "delayed healing",
    "partial paresis",
    "muscle stiffness",
    "Alopecia",
    "Obesity",
  ];
  // symptoms.forEach((symptom) => {
  //   buildDict[symptom] = "No";
  // });
  // console.log(buildDict);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      const response = await fetch("http://localhost:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });
      const data = await response.json();
      setPrediction(data);
    } catch (error) {
      console.error("Prediction error:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-3xl font-bold text-blue-800 mb-2">
          Diabetes Risk Assessment
        </h1>
        <p className="text-gray-600 mb-8">
          Early detection system using federated learning
        </p>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Input Form */}
          <div className="bg-white rounded-xl shadow-md p-6 col-span-1">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">
              Patient Information
            </h2>
            <form onSubmit={handleSubmit}>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700">
                    Age
                  </label>
                  <input
                    type="number"
                    min={0}
                    className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                    value={formData.Age}
                    onChange={(e) =>
                      setFormData({ ...formData, Age: e.target.value })
                    }
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700">
                    Gender
                  </label>
                  <select
                    className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                    value={formData.Gender}
                    onChange={(e) =>
                      setFormData({ ...formData, Gender: e.target.value })
                    }
                  >
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                  </select>
                </div>

                {/* Symptoms */}
                <div className="pt-4 border-t border-gray-200">
                  <h3 className="text-lg font-medium text-gray-800">
                    Symptoms
                  </h3>
                  {symptoms.map((symptom) => (
                    <div key={symptom} className="mt-3">
                      <label className="flex items-center">
                        <input
                          type="checkbox"
                          className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                          checked={formData[symptom] === "Yes"}
                          onChange={(e) =>
                            setFormData({
                              ...formData,
                              [symptom]: e.target.checked ? "Yes" : "No",
                            })
                          }
                        />
                        <span className="ml-2 text-sm text-gray-700 capitalize">
                          {symptom.replace(/([A-Z])/g, " $1").trim()}
                        </span>
                      </label>
                    </div>
                  ))}
                </div>

                <button
                  type="submit"
                  disabled={loading}
                  className={`w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 ${
                    loading ? "opacity-70 cursor-not-allowed" : ""
                  }`}
                >
                  {loading ? "Assessing..." : "Assess Risk"}
                </button>
              </div>
            </form>
          </div>

          {/* Results Panel */}
          <div className="lg:col-span-2 space-y-6">
            {prediction ? (
              <>
                <RiskMeter prediction={prediction} />

                <div className="bg-white rounded-xl shadow-md p-6">
                  <h2 className="text-xl font-semibold text-gray-800 mb-4">
                    Clinical Guidance
                  </h2>
                  <div className="space-y-3">
                    {prediction.clinical_actions.map((action, i) => (
                      <div key={i} className="flex items-start">
                        <div className="flex-shrink-0 h-5 w-5 text-blue-500 mt-0.5">
                          <svg
                            xmlns="http://www.w3.org/2000/svg"
                            viewBox="0 0 20 20"
                            fill="currentColor"
                          >
                            <path
                              fillRule="evenodd"
                              d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                              clipRule="evenodd"
                            />
                          </svg>
                        </div>
                        <p className="ml-3 text-sm text-gray-700">{action}</p>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="bg-white rounded-xl shadow-md p-6">
                  <h2 className="text-xl font-semibold text-gray-800 mb-4">
                    Key Risk Factors
                  </h2>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {prediction.feature_analysis.map((feature, i) => (
                      <FeatureCard key={i} feature={feature} />
                    ))}
                  </div>
                </div>
              </>
            ) : (
              <div className="bg-white rounded-xl shadow-md p-8 text-center">
                <svg
                  className="mx-auto h-12 w-12 text-gray-400"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                  />
                </svg>
                <h3 className="mt-2 text-lg font-medium text-gray-900">
                  No assessment yet
                </h3>
                <p className="mt-1 text-sm text-gray-500">
                  Submit patient information to evaluate diabetes risk.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
