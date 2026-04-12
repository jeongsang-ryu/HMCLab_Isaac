"""Category-shared sensor factories.

Sensors are parameterized by the robot prim path they attach to, so the same
factory can be reused across different robots and across multiple instances
(e.g. ego + opponent) within a single environment.

Isaac Lab sensor modules rely on pxr/usd which is only loaded after AppLauncher
bootstraps Kit. Keep these imports lazy — import inside the factory function
bodies, not at module top level.
"""
