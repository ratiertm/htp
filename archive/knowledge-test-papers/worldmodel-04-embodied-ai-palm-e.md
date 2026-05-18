# Embodied AI and Vision Language Action Models

Embodied AI extends large language models to physical agents that perceive sensory
input and execute actions in the real world. PaLM-E demonstrated that pretrained
vision language models can be fine tuned for robotic manipulation by treating
sensor observations and action commands as additional modalities of a unified
multimodal sequence model. The resulting system performs visual question answering,
long horizon planning, and motor control through a single transformer architecture.

This approach contrasts with traditional robotics pipelines that separate
perception, planning, and control into distinct modules. By training a unified
model on diverse multimodal data, embodied AI agents acquire general world
knowledge transferable across tasks and embodiments. Recent vision language action
models extend this paradigm with explicit action tokenization, enabling direct
prediction of low level motor commands from visual context. Embodied AI represents
a fundamental shift from disembodied language models toward situated agents that
ground their knowledge through interaction with physical environments, bridging
foundation models and traditional robotics.
