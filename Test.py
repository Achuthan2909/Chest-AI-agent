from r1_smolagent_rag import ChestXRayAgent

# Initialize the agent
agent = ChestXRayAgent()

# Test data
diagnosis = "Pneumonia"
confidence = 0.92
activations = {
    'upper_left': 0.512,
    'upper_middle': 0.544,
    'upper_right': 0.429,
    'middle_left': 0.538,
    'middle_middle': 0.522,
    'middle_right': 0.348,
    'lower_left': 0.390,
    'lower_middle': float('nan'),
    'lower_right': float('nan')
}

# Generate report
report = agent.generate_report(diagnosis, activations, confidence)
print("\n🔹 Generated Chest X-ray Report:\n", report)