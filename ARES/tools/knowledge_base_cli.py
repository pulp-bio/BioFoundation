#!/usr/bin/env python
# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Knowledge Base CLI Tool

Manage and query the ARES optimization knowledge base.

Usage:
    # List all entries
    python tools/knowledge_base_cli.py list

    # List entries by op type
    python tools/knowledge_base_cli.py list --op-type linear_int8

    # Search for entries matching shape
    python tools/knowledge_base_cli.py search --op-type linear_int8 --shape '{"M": 400, "N": 768}'

    # Show statistics
    python tools/knowledge_base_cli.py stats

    # Export to markdown
    python tools/knowledge_base_cli.py export --format markdown > kb_report.md

    # Export to JSON (subset)
    python tools/knowledge_base_cli.py export --format json --op-type linear_int8

    # Prune low-confidence entries
    python tools/knowledge_base_cli.py prune --min-confidence 0.6 --dry-run

    # Delete a specific entry by index
    python tools/knowledge_base_cli.py delete --index 5

    # Show negative results (what didn't work)
    python tools/knowledge_base_cli.py negatives

    # Lookup best config for a layer
    python tools/knowledge_base_cli.py lookup --op-type linear_int8 --shape '{"M": 400, "N": 768, "K": 192}'
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from codegen.optimization import KnowledgeBase


def cmd_list(kb: KnowledgeBase, args) -> int:
    """List entries in the knowledge base."""
    entries = kb.entries

    if args.op_type:
        entries = [e for e in entries if e.op_type == args.op_type]

    if not entries:
        print("No entries found.")
        return 0

    print(f"\n{'='*80}")
    print(f"Knowledge Base Entries ({len(entries)} total)")
    print(f"{'='*80}\n")

    print(f"{'#':<4} {'Op Type':<20} {'Description':<30} {'MACs/cyc':>10} {'Conf':>6}")
    print("-" * 75)

    for i, entry in enumerate(entries):
        desc = entry.description[:28] + ".." if len(entry.description or "") > 30 else (entry.description or "N/A")
        macs = f"{entry.performance.macs_per_cycle:.1f}" if entry.performance.macs_per_cycle else "N/A"
        print(f"{i:<4} {entry.op_type:<20} {desc:<30} {macs:>10} {entry.confidence:>5.0%}")

    return 0


def cmd_search(kb: KnowledgeBase, args) -> int:
    """Search for matching entries."""
    if not args.op_type:
        print("Error: --op-type required for search")
        return 1

    shape = args.shape or {}

    # Find matches
    matches = kb.lookup_all(args.op_type, shape, min_confidence=args.min_confidence)

    if not matches:
        print(f"No matching entries for {args.op_type} with shape {shape}")
        return 0

    print(f"\n{'='*80}")
    print(f"Search Results for {args.op_type}")
    print(f"Shape: {shape}")
    print(f"{'='*80}\n")

    for entry, score in matches[:10]:
        print(f"\n[Score: {score:.2f}] {entry.description or 'N/A'}")
        print(f"  Shape pattern: {entry.shape_pattern.pattern}")
        print(f"  Tile config: {entry.tile_config.config}")
        if entry.kernel_config.config:
            print(f"  Kernel config: {entry.kernel_config.config}")
        if entry.compile_flags.flags:
            print(f"  Compile flags: {entry.compile_flags.flags}")
        if entry.performance.macs_per_cycle:
            print(f"  MACs/cycle: {entry.performance.macs_per_cycle:.2f}")
        print(f"  Source: {entry.source}")
        print(f"  Confidence: {entry.confidence:.0%}")

    return 0


def cmd_lookup(kb: KnowledgeBase, args) -> int:
    """Lookup best config for a layer."""
    if not args.op_type:
        print("Error: --op-type required for lookup")
        return 1

    shape = args.shape or {}

    entry = kb.lookup(args.op_type, shape, min_confidence=args.min_confidence)

    if not entry:
        print(f"No matching config for {args.op_type} with shape {shape}")
        return 1

    print(f"\n{'='*60}")
    print(f"Best Config for {args.op_type}")
    print(f"{'='*60}\n")

    print(f"Description: {entry.description or 'N/A'}")
    print(f"\nShape pattern: {entry.shape_pattern.pattern}")
    print(f"\nTile config:")
    for k, v in entry.tile_config.config.items():
        print(f"  {k}: {v}")

    if entry.kernel_config.config:
        print(f"\nKernel config:")
        for k, v in entry.kernel_config.config.items():
            print(f"  {k}: {v}")

    if entry.compile_flags.flags:
        print(f"\nCompile flags:")
        for k, v in entry.compile_flags.flags.items():
            print(f"  {k}: {v}")

    print(f"\nPerformance:")
    if entry.performance.macs_per_cycle:
        print(f"  MACs/cycle: {entry.performance.macs_per_cycle:.2f}")
    if entry.performance.total_cycles:
        print(f"  Total cycles: {entry.performance.total_cycles:,}")
    if entry.performance.overlap_ratio:
        print(f"  Overlap ratio: {entry.performance.overlap_ratio:.0%}")

    print(f"\nMetadata:")
    print(f"  Source: {entry.source}")
    print(f"  Confidence: {entry.confidence:.0%}")
    if entry.test_network:
        print(f"  Test network: {entry.test_network}")

    return 0


def cmd_stats(kb: KnowledgeBase, args) -> int:
    """Show knowledge base statistics."""
    print(f"\n{'='*60}")
    print("Knowledge Base Statistics")
    print(f"{'='*60}\n")

    print(f"Version: {kb.version}")
    print(f"Last updated: {kb.last_updated}")
    print(f"Total entries: {len(kb.entries)}")
    print(f"Negative results: {len(kb.negative_results)}")
    print(f"Performance baselines: {len(kb.performance_baselines)}")
    print(f"Compile flag references: {len(kb.compile_flag_reference)}")

    # Entries by op type
    print("\nEntries by op type:")
    op_types = {}
    for entry in kb.entries:
        op_types[entry.op_type] = op_types.get(entry.op_type, 0) + 1

    for op_type in sorted(op_types.keys()):
        count = op_types[op_type]
        best = kb.get_best_entry(op_type)
        best_macs = f"{best.performance.macs_per_cycle:.1f}" if best and best.performance.macs_per_cycle else "N/A"
        print(f"  {op_type:<20} {count:>3} entries  (best: {best_macs} MACs/cyc)")

    # Confidence distribution
    print("\nConfidence distribution:")
    high = sum(1 for e in kb.entries if e.confidence >= 0.9)
    medium = sum(1 for e in kb.entries if 0.7 <= e.confidence < 0.9)
    low = sum(1 for e in kb.entries if e.confidence < 0.7)
    print(f"  High (≥90%):   {high}")
    print(f"  Medium (70-89%): {medium}")
    print(f"  Low (<70%):    {low}")

    # Sources
    print("\nSources:")
    sources = {}
    for entry in kb.entries:
        sources[entry.source] = sources.get(entry.source, 0) + 1
    for source in sorted(sources.keys()):
        print(f"  {source}: {sources[source]}")

    return 0


def cmd_export(kb: KnowledgeBase, args) -> int:
    """Export knowledge base."""
    if args.format == "markdown":
        print(kb.export_summary())
    elif args.format == "json":
        entries = kb.entries
        if args.op_type:
            entries = [e for e in entries if e.op_type == args.op_type]

        data = {
            "version": kb.version,
            "entries": [e.to_dict() for e in entries],
        }
        print(json.dumps(data, indent=2))
    else:
        print(f"Unknown format: {args.format}")
        return 1

    return 0


def cmd_negatives(kb: KnowledgeBase, args) -> int:
    """Show negative results."""
    if not kb.negative_results:
        print("No negative results recorded.")
        return 0

    print(f"\n{'='*60}")
    print("Negative Results (What Didn't Work)")
    print(f"{'='*60}\n")

    for i, neg in enumerate(kb.negative_results):
        print(f"\n[{i}] {neg.op_type}")
        print(f"    Attempted: {neg.attempted_config}")
        print(f"    Result: {neg.result}")
        if neg.notes:
            print(f"    Notes: {neg.notes}")
        if neg.source:
            print(f"    Source: {neg.source}")

    return 0


def cmd_prune(kb: KnowledgeBase, args) -> int:
    """Prune low-confidence entries."""
    to_remove = [
        e for e in kb.entries
        if e.confidence < args.min_confidence
    ]

    if not to_remove:
        print(f"No entries with confidence < {args.min_confidence:.0%}")
        return 0

    print(f"\nEntries to remove ({len(to_remove)}):")
    for entry in to_remove:
        print(f"  - {entry.op_type}: {entry.description} (conf: {entry.confidence:.0%})")

    if args.dry_run:
        print("\nDry run - no changes made.")
        return 0

    # Actually remove
    kb.entries = [e for e in kb.entries if e.confidence >= args.min_confidence]
    kb.save()
    print(f"\nRemoved {len(to_remove)} entries. KB now has {len(kb.entries)} entries.")

    return 0


def cmd_delete(kb: KnowledgeBase, args) -> int:
    """Delete a specific entry."""
    if args.index < 0 or args.index >= len(kb.entries):
        print(f"Invalid index: {args.index}. Valid range: 0-{len(kb.entries)-1}")
        return 1

    entry = kb.entries[args.index]
    print(f"\nEntry to delete:")
    print(f"  Op type: {entry.op_type}")
    print(f"  Description: {entry.description}")
    print(f"  Source: {entry.source}")

    if args.dry_run:
        print("\nDry run - no changes made.")
        return 0

    # Actually remove
    kb.entries.pop(args.index)
    kb.save()
    print(f"\nEntry deleted. KB now has {len(kb.entries)} entries.")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="ARES Knowledge Base CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # List command
    list_parser = subparsers.add_parser("list", help="List entries")
    list_parser.add_argument("--op-type", help="Filter by op type")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search for entries")
    search_parser.add_argument("--op-type", required=True, help="Operation type")
    search_parser.add_argument("--shape", type=json.loads, help="Shape as JSON")
    search_parser.add_argument("--min-confidence", type=float, default=0.0, help="Min confidence")

    # Lookup command
    lookup_parser = subparsers.add_parser("lookup", help="Lookup best config")
    lookup_parser.add_argument("--op-type", required=True, help="Operation type")
    lookup_parser.add_argument("--shape", type=json.loads, help="Shape as JSON")
    lookup_parser.add_argument("--min-confidence", type=float, default=0.0, help="Min confidence")

    # Stats command
    subparsers.add_parser("stats", help="Show statistics")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export knowledge base")
    export_parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    export_parser.add_argument("--op-type", help="Filter by op type (json only)")

    # Negatives command
    subparsers.add_parser("negatives", help="Show negative results")

    # Prune command
    prune_parser = subparsers.add_parser("prune", help="Prune low-confidence entries")
    prune_parser.add_argument("--min-confidence", type=float, default=0.7, help="Min confidence to keep")
    prune_parser.add_argument("--dry-run", "-n", action="store_true", help="Show what would be removed")

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a specific entry")
    delete_parser.add_argument("--index", type=int, required=True, help="Entry index to delete")
    delete_parser.add_argument("--dry-run", "-n", action="store_true", help="Don't actually delete")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Load knowledge base
    kb = KnowledgeBase()

    # Dispatch to command handler
    handlers = {
        "list": cmd_list,
        "search": cmd_search,
        "lookup": cmd_lookup,
        "stats": cmd_stats,
        "export": cmd_export,
        "negatives": cmd_negatives,
        "prune": cmd_prune,
        "delete": cmd_delete,
    }

    handler = handlers.get(args.command)
    if handler:
        return handler(kb, args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
