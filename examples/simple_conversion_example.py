"""
Simple working example: Converting AST and dataclass to PyTorch Geometric format.

This script provides practical examples you can run immediately.
"""

import torch
from torch_geometric.loader import DataLoader

# Import our converters
from src.gnn_vuln_detection.data_processing.graph_converter import (
    CodeSample,
    DataclassToGraphConverter,
    VulnerabilityDataset,
    create_vulnerability_dataset,
)


def simple_example_1():
    """Simple example: CodeSample to PyTorch Geometric Data."""
    print("=== Example 1: CodeSample to PyG Data ===")

    # Create a vulnerable code sample
    vulnerable_sample = CodeSample(
        code="char buf[10]; strcpy(buf, input);",  # Buffer overflow
        label=1,  # 1 = vulnerable
        language="c",
        cwe_id="CWE-120",
    )

    # Convert to PyG Data
    converter = DataclassToGraphConverter()
    data = converter.code_sample_to_pyg_data(vulnerable_sample)

    print("✅ Successfully converted to PyG Data:")
    print(f"   Node features: {data.x.shape}")
    print(f"   Edges: {data.edge_index.shape}")
    print(f"   Label: {data.y}")
    print(f"   CWE: {data.cwe_id}")

    return data


def simple_example_2():
    """Simple example: Multiple samples to dataset."""
    print("\\n=== Example 2: Multiple Samples to Dataset ===")

    # Create multiple samples
    samples = [
        CodeSample("int x = 5;", 0, "c"),  # Safe
        CodeSample("strcpy(buf, input);", 1, "c"),  # Vulnerable
        CodeSample("printf('hello');", 0, "c"),  # Safe
        CodeSample("gets(buffer);", 1, "c"),  # Vulnerable
    ]

    # Create dataset
    dataset = VulnerabilityDataset(samples)

    print(f"✅ Created dataset with {len(dataset)} samples")

    # Create data loader
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Get a batch
    batch = next(iter(loader))
    print(f"   Batch size: {batch.num_graphs}")
    print(f"   Total nodes: {batch.x.shape[0]}")
    print(f"   Labels: {batch.y}")

    return dataset, loader


def simple_example_3():
    """Simple example: Train/validation split."""
    print("\\n=== Example 3: Train/Validation Split ===")

    # Create more samples
    samples = []

    # Vulnerable patterns
    vulnerable_patterns = [
        "strcpy(dest, src);",
        "gets(buffer);",
        "sprintf(buf, format);",
        "system(command);",
    ]

    # Safe patterns
    safe_patterns = [
        "int x = 1;",
        "return 0;",
        "printf('test');",
        "x = y + z;",
    ]

    # Create samples
    for pattern in vulnerable_patterns:
        samples.append(CodeSample(pattern, 1, "c"))

    for pattern in safe_patterns:
        samples.append(CodeSample(pattern, 0, "c"))

    # Split into train/val/test
    train_ds, val_ds, test_ds = create_vulnerability_dataset(
        samples,
        train_split=0.6,
        val_split=0.2,
    )

    print("✅ Split dataset:")
    print(f"   Train: {len(train_ds)} samples")
    print(f"   Validation: {len(val_ds)} samples")
    print(f"   Test: {len(test_ds)} samples")

    return train_ds, val_ds, test_ds


def simple_example_4() -> None:
    """Simple example: Basic GNN forward pass."""
    print("\\n=== Example 4: GNN Forward Pass ===")

    try:
        from src.gnn_vuln_detection.models.gnn import create_vulnerability_detector

        # Create simple dataset
        samples = [
            CodeSample("int safe = 1;", 0, "c"),
            CodeSample("strcpy(buf, input);", 1, "c"),
        ]

        dataset = VulnerabilityDataset(samples)
        loader = DataLoader(dataset, batch_size=2)

        # Get input dimensions
        sample = dataset[0]
        input_dim = sample.x.shape[1]

        # Create model
        model = create_vulnerability_detector(
            model_type="gcn",
            input_dim=input_dim,
            hidden_dim=32,
            num_classes=2,
        )

        print(f"✅ Created GCN model with {input_dim} input features")

        # Forward pass
        model.eval()
        batch = next(iter(loader))

        with torch.no_grad():
            output = model(batch)
            predictions = output.argmax(dim=1)

        print(f"   Input shape: {batch.x.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   True labels: {batch.y.tolist()}")
        print(f"   Predictions: {predictions.tolist()}")

    except ImportError:
        print("⚠️  GNN models not available. Install torch_geometric.")
    except Exception as e:
        print(f"⚠️  Error: {e}")


def demonstrate_data_format() -> None:
    """Show the expected PyTorch Geometric data format."""
    print("\\n=== PyTorch Geometric Data Format ===")

    # Manual example of PyG Data structure
    print("Expected format for vulnerability detection:")
    print("""
    data = Data(
        x=torch.tensor([[...], [...]]),      # Node features [num_nodes, num_features]
        edge_index=torch.tensor([[0, 1],    # Edge connectivity [2, num_edges]
                                 [1, 2]]),
        y=torch.tensor([1]),                 # Graph label [1] (0=safe, 1=vulnerable)
        batch=torch.tensor([0, 0, 1, 1])    # Which graph each node belongs to
    )
    """)

    # Create actual example
    sample = CodeSample("int x = 1;", 0, "c")
    converter = DataclassToGraphConverter()
    data = converter.code_sample_to_pyg_data(sample)

    print("\\nActual converted data:")
    print(f"   x.shape: {data.x.shape}")
    print(f"   edge_index.shape: {data.edge_index.shape}")
    print(f"   y: {data.y}")
    print(f"   Available attributes: {list(data.keys)}")


def main() -> None:
    """Run all examples."""
    print("🚀 Simple PyTorch Geometric Conversion Examples\\n")

    try:
        simple_example_1()
        simple_example_2()
        simple_example_3()
        simple_example_4()
        demonstrate_data_format()

        print("\\n✅ All examples completed successfully!")
        print("\\n📝 Key takeaways:")
        print("1. CodeSample dataclass easily converts to PyG Data")
        print("2. Multiple samples create PyG Datasets")
        print("3. DataLoader handles batching automatically")
        print("4. GNN models can process the converted data directly")

    except Exception as e:
        print(f"\\n❌ Error: {e}")
        print("Make sure dependencies are installed:")
        print("pip install torch torch_geometric tree_sitter numpy")


if __name__ == "__main__":
    main()
