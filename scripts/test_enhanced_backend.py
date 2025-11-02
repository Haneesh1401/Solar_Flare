#!/usr/bin/env python3
"""
Test script for the Enhanced Solar Flare Prediction Backend API
Tests all new endpoints and functionality
"""

import requests
import json
import time
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

def test_health_endpoint(base_url="http://localhost:8000"):
    """Test health check endpoint"""
    print("🩺 Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            data = response.json()
            print("✅ Health check passed")
            print(f"   Status: {data['status']}")
            print(f"   Models loaded: {data['models_loaded']}")
            print(f"   Cache status: {data['cache_status']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {str(e)}")
        return False

def test_single_prediction(base_url="http://localhost:8000"):
    """Test single prediction endpoint"""
    print("\n🔮 Testing single prediction...")
    try:
        # Test data
        test_data = {
            "flux": 1.5,
            "month": 6,
            "day": 15,
            "hour": 12,
            "day_of_year": 166
        }

        response = requests.post(f"{base_url}/predict", json=test_data)
        if response.status_code == 200:
            data = response.json()
            print("✅ Single prediction successful")
            print(f"   Model used: {data['model_used']}")
            print(f"   Prediction: {data['prediction']}")
            print(f"   Confidence: {data['confidence']:.3f}")
            print(f"   Processing time: {data['processing_time']:.3f}s")
            return True
        else:
            print(f"❌ Single prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Single prediction error: {str(e)}")
        return False

def test_batch_prediction(base_url="http://localhost:8000"):
    """Test batch prediction endpoint"""
    print("\n📦 Testing batch prediction...")
    try:
        # Test data
        test_data = {
            "inputs": [
                {"flux": 1.0, "month": 6, "day": 15, "hour": 10, "day_of_year": 166},
                {"flux": 2.5, "month": 6, "day": 15, "hour": 14, "day_of_year": 166},
                {"flux": 0.8, "month": 6, "day": 15, "hour": 18, "day_of_year": 166}
            ]
        }

        response = requests.post(f"{base_url}/predict/batch", json=test_data)
        if response.status_code == 200:
            data = response.json()
            print("✅ Batch prediction successful")
            print(f"   Predictions count: {len(data['predictions'])}")
            print(f"   Average confidence: {data['average_confidence']:.3f}")
            print(f"   Total processing time: {data['total_processing_time']:.3f}s")

            # Show individual predictions
            for i, pred in enumerate(data['predictions']):
                print(f"   Prediction {i+1}: {pred['prediction']} (confidence: {pred['confidence']:.3f})")

            return True
        else:
            print(f"❌ Batch prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Batch prediction error: {str(e)}")
        return False

def test_model_management(base_url="http://localhost:8000"):
    """Test model management endpoints"""
    print("\n🔧 Testing model management...")

    try:
        # Test list models
        response = requests.get(f"{base_url}/models")
        if response.status_code == 200:
            models = response.json()
            print(f"✅ Found {len(models)} models in registry")

            for model in models[:3]:  # Show first 3 models
                print(f"   - {model['name']} (v{model['version']}) - {model['type']}")

            if len(models) > 3:
                print(f"   ... and {len(models) - 3} more models")

            return True
        else:
            print(f"❌ List models failed: {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ Model management error: {str(e)}")
        return False

def test_performance_monitoring(base_url="http://localhost:8000"):
    """Test performance monitoring endpoint"""
    print("\n📊 Testing performance monitoring...")
    try:
        response = requests.get(f"{base_url}/monitoring/performance")
        if response.status_code == 200:
            data = response.json()
            print("✅ Performance monitoring successful")
            print(f"   Model: {data['model']}")
            print(f"   Accuracy: {data['metrics']['accuracy']:.4f}")
            print(f"   F1 Score: {data['metrics']['f1_score']:.4f}")
            return True
        else:
            print(f"❌ Performance monitoring failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Performance monitoring error: {str(e)}")
        return False

def test_model_comparison(base_url="http://localhost:8000"):
    """Test model comparison endpoint"""
    print("\n⚖️ Testing model comparison...")
    try:
        response = requests.get(f"{base_url}/monitoring/models/compare")
        if response.status_code == 200:
            data = response.json()
            print("✅ Model comparison successful")
            print(f"   Compared {len(data['comparison'])} models")

            for model_name, info in data['comparison'].items():
                if 'error' not in info:
                    metrics = info['metrics']
                    print(f"   - {model_name}: F1={metrics['f1_score']:.4f}, Acc={metrics['accuracy']:.4f}")
                else:
                    print(f"   - {model_name}: Error - {info['error']}")

            return True
        else:
            print(f"❌ Model comparison failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Model comparison error: {str(e)}")
        return False

def test_realtime_prediction(base_url="http://localhost:8000"):
    """Test real-time prediction endpoint"""
    print("\n🌐 Testing real-time prediction...")
    try:
        response = requests.get(f"{base_url}/predict/realtime")
        if response.status_code == 200:
            data = response.json()
            print("✅ Real-time prediction successful")
            print(f"   Time: {data['time']}")
            print(f"   Flux: {data['flux']}")
            print(f"   Prediction: {data['prediction']['prediction']}")
            print(f"   Confidence: {data['prediction']['confidence']:.3f}")
            return True
        else:
            print(f"❌ Real-time prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Real-time prediction error: {str(e)}")
        return False

def test_prometheus_metrics(base_url="http://localhost:8000"):
    """Test Prometheus metrics endpoint"""
    print("\n📈 Testing Prometheus metrics...")
    try:
        response = requests.get(f"{base_url}/metrics")
        if response.status_code == 200:
            metrics_text = response.text
            print("✅ Prometheus metrics available")

            # Count metric lines
            lines = metrics_text.strip().split('\n')
            metric_lines = [line for line in lines if line and not line.startswith('#')]
            print(f"   Found {len(metric_lines)} metrics")

            # Show some example metrics
            sample_metrics = [line for line in metric_lines if 'solar_flare' in line][:5]
            for metric in sample_metrics:
                print(f"   - {metric}")

            return True
        else:
            print(f"❌ Prometheus metrics failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Prometheus metrics error: {str(e)}")
        return False

def test_legacy_compatibility(base_url="http://localhost:8000"):
    """Test legacy endpoint for backward compatibility"""
    print("\n🔄 Testing legacy compatibility...")
    try:
        # Test data
        test_data = {
            "flux": 1.5,
            "month": 6,
            "day": 15
        }

        response = requests.post(f"{base_url}/predict/legacy", json=test_data)
        if response.status_code == 200:
            data = response.json()
            print("✅ Legacy endpoint working")
            print(f"   Prediction: {data['predicted_flare_class']}")
            return True
        else:
            print(f"❌ Legacy endpoint failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Legacy endpoint error: {str(e)}")
        return False

def run_all_tests(base_url="http://localhost:8000"):
    """Run all tests"""
    print("🚀 Starting Enhanced Backend API Tests")
    print("=" * 50)

    tests = [
        ("Health Check", test_health_endpoint),
        ("Single Prediction", test_single_prediction),
        ("Batch Prediction", test_batch_prediction),
        ("Model Management", test_model_management),
        ("Performance Monitoring", test_performance_monitoring),
        ("Model Comparison", test_model_comparison),
        ("Real-time Prediction", test_realtime_prediction),
        ("Prometheus Metrics", test_prometheus_metrics),
        ("Legacy Compatibility", test_legacy_compatibility)
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func(base_url)
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {str(e)}")
            results.append((test_name, False))

    # Summary
    print(f"\n{'='*50}")
    print("📋 TEST SUMMARY")
    print('='*50)

    passed = 0
    failed = 0

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
        else:
            failed += 1

    print('='*50)
    print(f"Total Tests: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {(passed/len(results)*100):.1f}%")

    if failed == 0:
        print("\n🎉 All tests passed! Enhanced backend is working perfectly.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the implementation.")

    return failed == 0

if __name__ == "__main__":
    # Allow custom base URL
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

    print(f"Testing against: {base_url}")
    print(f"Timestamp: {datetime.now().isoformat()}")

    success = run_all_tests(base_url)

    if success:
        print("\n✅ Enhanced Solar Flare Prediction Backend is ready for production!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please review and fix the issues.")
        sys.exit(1)
