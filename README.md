# AI-Generated Documentation Systems: Evaluation and Mitigation Strategies

[![Course](https://img.shields.io/badge/Course-SW%20403%3A%20AI%20in%20Modern%20Software-blue)](https://)

This repository contains the research and development for the "AI in Modern Software" (SW 403) course project, focusing on the challenges and opportunities of AI-generated documentation in critical domains like software engineering and healthcare.

## 1. Project Description

The proliferation of Large Language Models (LLMs) presents a significant opportunity to automate the costly and time-consuming process of documentation in both software engineering (especially for legacy systems) and healthcare (clinical notes). However, this automation is fraught with risks, including factual inaccuracies, hallucinations, and omissions, which can have severe consequences in these safety-critical fields.

This project investigates the current state of AI-generated documentation, aiming to develop and evaluate a prototype system. The core of the project is to analyze failure modes (like hallucinations) and propose frameworks for evaluating documentation quality, balancing the need for efficiency with the demand for reliability and trustworthiness.

## 2. Motivation

* **Software Engineering:** Code documentation is vital for maintenance and knowledge transfer, yet it's often neglected. In legacy systems (e.g., COBOL, MUMPS), where original developers are unavailable, documentation is critical. LLMs struggle with these less-common languages and complex code, often producing incomplete or "hallucinated" explanations.
* **Healthcare:** Clinicians spend 34-55% of their time on documentation, leading to burnout and massive opportunity costs. While AI can significantly reduce this burden, an error or omission in a clinical note could directly compromise patient safety.

This research aims to bridge the gap between the potential efficiency gains of AI and the rigorous quality and safety standards required in these domains.

## 3. Research Objectives

Based on our initial literature review, this project seeks to answer the following research questions:

1.  **RQ1:** How can evaluation frameworks for AI-generated documentation be unified across domains (software vs. healthcare) while remaining sensitive to domain-specific priorities (e.g., technical accuracy vs. patient safety)?
2.  **RQ2:** What systematic patterns characterize AI documentation failures (hallucinations, omissions), and can these patterns be predicted or mitigated through targeted interventions (like structured prompting or RAG)?
3.  **RQ3:** How should human-in-the-loop (HITL) processes be structured to balance efficiency gains with the necessary quality assurance and accountability in critical systems?
4.  **RQ4:** What regulatory and governance frameworks are needed to balance innovation with safety and accountability for AI documentation in regulated industries?
5.  **RQ5:** Do AI documentation systems equitably serve diverse user populations (e.g., novice developers, non-native speakers), and how can biases in training data be identified and mitigated?

## 4. Project Phases
This project is structured into three main phases, aligned with the course timeline.
<table width="100%">
  <thead>
    <tr>
      <th align="left">Phase</th>
      <th align="left">Timeline</th>
      <th align="left">Deliverables</th>
      <th align="left" width="100%">Focus & Guidance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td valign="top" nowrap><strong>Phase 1</strong></td>
      <td valign="top" nowrap>Week 6</td>
      <td valign="top" nowrap>
        • Topic proposal (2–3 pages)<br>
        • Literature review summary<br>
        • Research question(s)
      </td>
      <td valign="top" width="100%">
        <strong>[COMPLETED]</strong> <br>Selected the task of evaluating AI-generated documentation. Conducted a literature review to identify the research gap, specifically the trade-off between efficiency and reliability, and defined our core research questions.
      </td>
    </tr>
    <tr>
      <td valign="top" nowrap><strong>Phase 2</strong></td>
      <td valign="top" nowrap>Week 10</td>
      <td valign="top" nowrap>
        • Prototype implementation<br>
        • Experimental setup & dataset<br>
        • Early results
      </td>
      <td valign="top" width="100%">
        <strong>[IN PROGRESS]</strong> <br>Develop a minimal viable prototype for either generating or evaluating AI documentation (e.g., for a specific legacy codebase or set of mock clinical notes). Collect data, run initial experiments, and critically analyze outputs. Document all errors, omissions, and hallucinations.
      </td>
    </tr>
    <tr>
      <td valign="top" nowrap><strong>Phase 3</strong></td>
      <td valign="top" nowrap>Week 14</td>
      <td valign="top" nowrap>
        • Complete Prototype<br>
        • Full research report (6–8 pages)<br>
        • Final demo and presentation<br>
        • Reflection
      </td>
      <td valign="top" width="100%">
        Write and present a full research paper summarizing our background, methods, and results from Phase 2. Include a thorough hallucination analysis and ethical aspects. The reflection will address whether our prototype/framework was able to enhance or better evaluate SOTA models.
      </td>
    </tr>
  </tbody>
</table>

## 5. Getting Started
*(This section will be updated once the prototype is developed.)*

To get a local copy up and running, follow these simple steps.

### Prerequisites

* List any required software and libraries
    ```sh
    pip install ...
    ```

### Installation

1.  Clone the repo
    ```sh
    git clone [https://github.com/your_username/your_repository.git](https://github.com/your_username/your_repository.git)
    ```
2.  Install packages
    ```sh
    ...
    ```

## 6. Usage

*(This section will be updated with instructions on how to run the prototype and experiments.)*

```python
# Add code examples here
````

## 7\. Team & Acknowledgements

  * **Ahmed Mostafa** (202201114)
  * **Ahmed Emad** (202202231)
  * **Seif Eldin** (202200973)

We would like to thank our supervisor, **Prof. Doaa Shawky**, for her guidance and support on this project.
