# Audit Evidence, Traceability, and Retention Policy

**Policy ID:** AUD-001  
**Version:** 2.3  
**Effective date:** 2026-01-01  
**Owner:** Internal Control Assurance

## Required trace

The close record must identify source files, versions, transformations, calculations, agent and tool actions, human decisions, model and prompt versions where applicable, errors, retries, and final outputs.

## Reproducibility

A reviewer must be able to reproduce material calculations using retained inputs. Generated narrative without underlying calculation and source evidence is insufficient.

## Agent records

Agent logs must contain structured actions and outputs but must not depend on private chain-of-thought. Task IDs, timestamps, selected skills, tool calls, evidence references, and decision summaries are required.

## Retention and access

Access follows least privilege and portfolio scope. Evidence must not be copied across portfolios without authorization. Rejected, failed, and canceled tasks remain visible in the audit history.
