<!--
Sync Impact Report
Version change: template -> 1.0.0
Modified principles:
- Template Principle 1 -> I. Research-First Modular Architecture
- Template Principle 2 -> II. Python Scientific Stack With Required Landmarking
- Template Principle 3 -> III. Multimodal Feature Grounding
- Template Principle 4 -> IV. Temporal Transformer Regression
- Template Principle 5 -> V. MVP-First, Testable Research Delivery
Added sections:
- Research and Technical Constraints
- Development Workflow and Quality Gates
Removed sections:
- Placeholder Section 2
- Placeholder Section 3
Templates requiring updates:
- .specify/templates/plan-template.md: updated
- .specify/templates/spec-template.md: updated
- .specify/templates/tasks-template.md: updated
- .specify/templates/commands/*.md: not present
- AGENTS.md: updated
Follow-up TODOs: None
-->
# Real-time Cognitive State Estimation from Video Constitution

## Core Principles

### I. Research-First Modular Architecture

The project MUST prioritize reproducible research and modular implementation.
Video input, MediaPipe landmark extraction, multimodal feature extraction,
temporal windowing, PyTorch modeling, inference, and evaluation MUST remain
separate modules with explicit interfaces. Research experiments MUST be
reproducible from declared configuration, dependency versions, seeds where
applicable, and saved artifacts. New research ideas MUST integrate through
module contracts rather than bypassing the pipeline structure.

Rationale: Continuous cognitive state estimation is experimental; modular
research code lets the project compare methods without rewriting the system.

### II. Python Scientific Stack With Required Landmarking

The implementation MUST be Python based. MediaPipe MUST be the default and MVP
method for face and pose landmark extraction. PyTorch MUST be used for trainable
temporal models. Alternative landmarking or modeling libraries MAY be added only
as documented research extensions after the MVP baseline is preserved.

Rationale: A focused stack reduces integration risk and keeps experiments
comparable across feature extraction, training, and inference.

### III. Multimodal Feature Grounding

Features MUST represent multiple behavioral modalities: eyes, face, head pose,
and posture. Feature schemas MUST document units, coordinate frames, temporal
sampling assumptions, normalization, and missing-landmark handling. MVP features
MUST be derived from interpretable landmark geometry or documented transforms of
those landmarks, not an opaque video-only representation.

Rationale: Cognitive state scores need interpretable behavioral signals so
research findings can be inspected, challenged, and extended.

### IV. Temporal Transformer Regression

The main temporal model MUST be a PyTorch Temporal Transformer Encoder operating
over ordered feature windows. LSTM, GRU, or other recurrent models MAY be used
for baselines or ablation studies, but MUST NOT replace the primary model
without a constitutional amendment. The project MUST output continuous
regression scores for Fatigue Score, Attention Score, Stress Score, and
Engagement Score with documented ranges and normalization. Binary classification
MUST NOT be the primary project target.

Rationale: The project goal is continuous cognitive state estimation from time
varying behavior, not simple category detection.

### V. MVP-First, Testable Research Delivery

The first implementation milestone MUST be the smallest end-to-end pipeline that
ingests webcam or video input, extracts landmarks, computes multimodal features,
runs temporal transformer inference, and emits all four continuous scores. Code
MUST be clean, typed where practical, testable, and separated by module
responsibility. Code-affecting changes MUST include relevant tests for feature
shapes, missing landmark behavior, temporal batching, model input/output
contracts, and score schema, unless a research spike explicitly documents why
tests are deferred.

Rationale: The MVP creates a stable research baseline while leaving room for
future datasets, labels, features, metrics, and model variants.

## Research and Technical Constraints

- The project scope is real-time cognitive state estimation from webcam or video
  input; offline experiments MAY exist only when they support the real-time
  pipeline.
- The four required outputs are Fatigue Score, Attention Score, Stress Score,
  and Engagement Score. Any feature that affects outputs MUST preserve all four
  scores unless the constitution is amended.
- Plans that modify runtime inference MUST specify target latency, frame rate,
  or an explicit non-real-time research scope.
- Plans that modify feature extraction MUST identify affected modalities from
  eyes, face, head pose, and posture.
- Plans that modify modeling MUST keep the Temporal Transformer Encoder as the
  primary model and document any baseline or ablation model separately.
- Data, feature, and model artifacts MUST be versioned or named so experiments
  can be traced back to configuration and code.

## Development Workflow and Quality Gates

- Specifications MUST describe whether a change affects video input, landmark
  extraction, feature engineering, temporal modeling, score outputs, evaluation,
  or real-time inference.
- Implementation plans MUST pass the Constitution Check before research design
  work begins and again after design artifacts are produced.
- Task plans MUST preserve an MVP-first delivery order: setup, shared pipeline
  foundations, the first end-to-end scoring path, then research extensions.
- Reviews MUST verify module separation, Python stack compliance, MediaPipe and
  PyTorch usage, transformer-first modeling, continuous regression outputs, and
  focused tests.
- Any deliberate violation MUST be recorded in the plan Complexity Tracking
  table with the rejected simpler alternative and a migration path back to the
  constitution.

## Governance

This constitution supersedes conflicting project guidance. Amendments require a
documented change to this file, a Sync Impact Report, updates to affected
templates or runtime guidance, and reviewer confirmation that active specs and
plans remain consistent.

Versioning follows semantic versioning:
- MAJOR for removing or redefining a core principle or changing required outputs,
  primary model class, or mandatory technology stack.
- MINOR for adding principles, required sections, or materially expanding
  governance and quality gates.
- PATCH for clarifications, wording fixes, or non-semantic refinements.

Compliance is reviewed at specification, planning, task generation, and code
review time. When a research extension needs temporary deviation, the plan MUST
state the hypothesis, expected evidence, affected modules, and rollback or
promotion criteria.

**Version**: 1.0.0 | **Ratified**: 2026-05-08 | **Last Amended**: 2026-05-08
