# Swarm PyBullet

This repository is Swarm Subnet's fork of PyBullet/Bullet3.

The fork is maintained for Swarm benchmark and validator workloads that depend heavily on CPU rendering and fast simulator turnaround. In addition to the upstream physics engine and Python bindings, this fork carries Swarm-specific rendering and loading improvements, including:

- a dedicated depth-only rendering path (`ER_DEPTH_ONLY`)
- frustum culling for depth-only TinyRenderer passes
- per-triangle far-plane culling for depth-only rendering
- OpenMP acceleration for the TinyRenderer depth-only face loop
- simulator loading improvements in the shared-memory path

The Python import surface remains `import pybullet`, but the published distribution name for this fork is `swarm-bullet3`.

## Why this fork exists

Swarm benchmark spends significant time in repeated simulator setup and depth rendering. The upstream package is a strong base, but these workloads benefit from renderer-specific optimizations that reduce evaluation latency and improve throughput in our robotics benchmark environment.

## Relationship to upstream

This fork is based on the Bullet Physics SDK and PyBullet Python bindings.

Upstream project: https://github.com/bulletphysics/bullet3
Fork repository: https://github.com/swarm-subnet/bullet3-swarmfork

## License

Bullet and PyBullet are distributed under the zlib license. This fork keeps that license.
