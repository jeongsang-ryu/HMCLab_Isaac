"""Minimal PhysX Vehicle SDK box-car validation demo.

Builds a single 4-wheeled vehicle from raw USD + PhysxVehicle* schemas
(no Articulation), drives it forward with sinusoidal steering, and
records a chase cam. Used to validate the Vehicle SDK code path before
wrapping it for our RL environments.

Z-up world (vertical=+Z, longitudinal=+X, side=+Y).
DRIVE_BASIC mode (no engine/gears/clutch — direct wheel torque from accelerator).

Run:
    OMNI_KIT_ACCEPT_EULA=YES \
      /home/js/anaconda3/envs/hmclab_test/bin/python \
      scripts/physx_vehicle_box_demo.py --steps 1500
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=1500)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--label", type=str, default="boxcar")
parser.add_argument("--accel", type=float, default=0.5, help="accelerator 0..1")
parser.add_argument("--steer-amp", type=float, default=0.4, dest="steer_amp")
parser.add_argument("--steer-freq", type=float, default=0.2, dest="steer_freq")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = False
args.enable_cameras = True
launcher = AppLauncher(args)
sim_app = launcher.app

import isaaclab.sim as sim_utils  # noqa: E402

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)
from hmclab_isaac.utils.capture import ChaseCamRecorder  # noqa: E402

OUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "Itutorial", "outputs"
)


def build_vehicle(stage, vehicle_path: str, init_xy=(0.0, 0.0)):
    """Author a 4-wheel PhysX Vehicle (DRIVE_BASIC) with Z-up axes.

    Returns: (vehicle_prim, controller_api).
    """
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, PhysxSchema
    from omni.physx.scripts.physicsUtils import (
        add_collision_to_collision_group,
        add_physics_material_to_prim,
    )
    from omni.physx.scripts.utils import set_custom_metadata

    root = "/World"

    # --- materials & friction tables ---
    tarmac_path = f"{root}/TarmacMaterial"
    UsdShade.Material.Define(stage, tarmac_path)
    tarmac = UsdPhysics.MaterialAPI.Apply(stage.GetPrimAtPath(tarmac_path))
    tarmac.CreateStaticFrictionAttr(0.9)
    tarmac.CreateDynamicFrictionAttr(0.7)
    tarmac.CreateRestitutionAttr(0.0)
    PhysxSchema.PhysxMaterialAPI.Apply(tarmac.GetPrim())

    fric_table_path = f"{root}/SummerTireFrictionTable"
    fric_table = PhysxSchema.PhysxVehicleTireFrictionTable.Define(stage, fric_table_path)
    fric_table.CreateGroundMaterialsRel().AddTarget(tarmac_path)
    fric_table.CreateFrictionValuesAttr([0.85])

    # --- collision groups ---
    grp_chassis = f"{root}/VehicleChassisGroup"
    grp_wheel = f"{root}/VehicleWheelGroup"
    grp_query = f"{root}/VehicleGroundQueryGroup"
    grp_ground = f"{root}/GroundSurfaceGroup"
    g_chassis = UsdPhysics.CollisionGroup.Define(stage, grp_chassis)
    g_chassis.CreateFilteredGroupsRel().AddTarget(grp_query)
    g_wheel = UsdPhysics.CollisionGroup.Define(stage, grp_wheel)
    g_wheel.CreateFilteredGroupsRel().AddTarget(grp_query)
    g_wheel.CreateFilteredGroupsRel().AddTarget(grp_ground)
    g_query = UsdPhysics.CollisionGroup.Define(stage, grp_query)
    g_query.CreateFilteredGroupsRel().AddTarget(grp_chassis)
    g_query.CreateFilteredGroupsRel().AddTarget(grp_wheel)
    g_ground = UsdPhysics.CollisionGroup.Define(stage, grp_ground)
    g_ground.CreateFilteredGroupsRel().AddTarget(grp_ground)
    g_ground.CreateFilteredGroupsRel().AddTarget(grp_wheel)

    # --- shared wheel / suspension / tire / drive prototypes ---
    wheel_radius = 0.35
    wheel_width = 0.18
    vehicle_mass = 1500.0
    g_mag = 9.81

    wheel_path = f"{root}/SharedWheel"
    wheel_prim = UsdGeom.Scope.Define(stage, wheel_path).GetPrim()
    wheel = PhysxSchema.PhysxVehicleWheelAPI.Apply(wheel_prim)
    wheel.CreateRadiusAttr(wheel_radius)
    wheel.CreateWidthAttr(wheel_width)
    wheel.CreateMassAttr(20.0)
    wheel.CreateMoiAttr(1.225)
    wheel.CreateDampingRateAttr(0.25)

    sprung_mass = vehicle_mass / 4
    spring_strength = 45000.0
    max_droop = (sprung_mass * g_mag) / spring_strength
    travel = 2.0 * max_droop
    wheel_rest_susp = travel - max_droop

    susp_path = f"{root}/SharedSuspension"
    susp_prim = UsdGeom.Scope.Define(stage, susp_path).GetPrim()
    susp = PhysxSchema.PhysxVehicleSuspensionAPI.Apply(susp_prim)
    susp.CreateSpringStrengthAttr(spring_strength)
    susp.CreateSpringDamperRateAttr(4500.0)
    susp.CreateTravelDistanceAttr(travel)

    tire_rest_load = sprung_mass * g_mag
    front_tire_path = f"{root}/SharedFrontTire"
    rear_tire_path = f"{root}/SharedRearTire"
    for tp, lat in ((front_tire_path, 17.0), (rear_tire_path, 25.0)):
        prim = UsdGeom.Scope.Define(stage, tp).GetPrim()
        tire = PhysxSchema.PhysxVehicleTireAPI.Apply(prim)
        tire.CreateLateralStiffnessGraphAttr(Gf.Vec2f(2.0, lat * tire_rest_load))
        tire.CreateLongitudinalStiffnessAttr(5000.0)
        tire.CreateCamberStiffnessAttr(0.0)
        tire.CreateFrictionVsSlipGraphAttr(
            [Gf.Vec2f(0.0, 1.0), Gf.Vec2f(0.1, 1.0), Gf.Vec2f(1.0, 1.0)]
        )
        tire.CreateFrictionTableRel().AddTarget(fric_table_path)

    # DRIVE_BASIC: applied as PhysxVehicleDriveBasicAPI directly on the vehicle
    # prim (no separate engine/gears/clutch).

    # --- vehicle root ---
    chassis_half_height = 0.45
    chassis_dist_to_ground = 0.3
    chassis_center_to_ground = chassis_half_height + chassis_dist_to_ground
    com_to_ground = 0.5
    com_offset_z = com_to_ground - chassis_center_to_ground

    vx, vy = init_xy
    init_z = chassis_center_to_ground

    veh = UsdGeom.Xform.Define(stage, vehicle_path)
    veh.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionFloat).Set(
        Gf.Vec3f(float(vx), float(vy), float(init_z))
    )
    veh.AddOrientOp(precision=UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(1, 0, 0, 0))
    veh_prim = veh.GetPrim()

    set_custom_metadata(veh_prim, PhysxSchema.Tokens.referenceFrameIsCenterOfMass, False)
    UsdPhysics.RigidBodyAPI.Apply(veh_prim)

    # box dim: forward=X (4.5), side=Y (1.8), up=Z (0.9)
    box = Gf.Vec3f(4.5, 1.8, 0.9)
    massAPI = UsdPhysics.MassAPI.Apply(veh_prim)
    massAPI.CreateMassAttr(vehicle_mass)
    massAPI.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, com_offset_z))
    massAPI.CreateDiagonalInertiaAttr(
        Gf.Vec3f(
            (box[1] * box[1]) + (box[2] * box[2]),
            (box[0] * box[0]) + (box[2] * box[2]),
            (box[0] * box[0]) + (box[1] * box[1]),
        )
        * (1.0 / 12.0)
        * vehicle_mass
    )
    massAPI.CreatePrincipalAxesAttr(Gf.Quatf(1, 0, 0, 0))

    rb = PhysxSchema.PhysxRigidBodyAPI.Apply(veh_prim)
    rb.CreateDisableGravityAttr(True)
    rb.CreateSleepThresholdAttr(0)
    rb.CreateStabilizationThresholdAttr(0)

    vehAPI = PhysxSchema.PhysxVehicleAPI.Apply(veh_prim)
    vehAPI.CreateVehicleEnabledAttr(True)
    vehAPI.CreateSubStepThresholdLongitudinalSpeedAttr(5.0)
    vehAPI.CreateLowForwardSpeedSubStepCountAttr(3)
    vehAPI.CreateHighForwardSpeedSubStepCountAttr(1)
    vehAPI.CreateMinPassiveLongitudinalSlipDenominatorAttr(4.0)
    vehAPI.CreateMinActiveLongitudinalSlipDenominatorAttr(0.1)
    vehAPI.CreateMinLateralSlipDenominatorAttr(1.0)

    # DRIVE_BASIC: peak torque applied to driven wheels scaled by accelerator
    drive_basic = PhysxSchema.PhysxVehicleDriveBasicAPI.Apply(veh_prim)
    drive_basic.CreatePeakTorqueAttr(2000.0)

    # Controller (runtime input)
    ctrl = PhysxSchema.PhysxVehicleControllerAPI.Apply(veh_prim)
    ctrl.CreateAcceleratorAttr(0.0)
    ctrl.CreateBrake0Attr(0.0)
    ctrl.CreateBrake1Attr(0.0)
    ctrl.CreateSteerAttr(0.0)

    # Brakes
    b0 = PhysxSchema.PhysxVehicleBrakesAPI.Apply(veh_prim, PhysxSchema.Tokens.brakes0)
    b0.CreateMaxBrakeTorqueAttr(3600.0)

    # Steering on front wheels (indices 0,1)
    steer = PhysxSchema.PhysxVehicleSteeringAPI.Apply(veh_prim)
    steer.CreateWheelsAttr([0, 1])
    steer.CreateMaxSteerAngleAttr(0.55)

    # Differential: drive on rear wheels (2,3) — RWD
    diff = PhysxSchema.PhysxVehicleMultiWheelDifferentialAPI.Apply(veh_prim)
    diff.CreateWheelsAttr([2, 3])
    diff.CreateTorqueRatiosAttr([0.5, 0.5])
    diff.CreateAverageWheelSpeedRatiosAttr([0.5, 0.5])

    # --- wheel attachments ---
    # In Z-up: forward=X, side=Y, up=Z. Wheel cylinder spins about its width
    # axis = side (Y).
    wheelbase_half = 1.6  # X
    track_half = 0.8  # Y
    wheel_rest_z = wheel_radius - chassis_center_to_ground  # local-frame
    susp_frame_z = wheel_rest_z + wheel_rest_susp

    wheel_specs = [
        ("FrontLeft", +wheelbase_half, +track_half, front_tire_path, 0),
        ("FrontRight", +wheelbase_half, -track_half, front_tire_path, 1),
        ("RearLeft", -wheelbase_half, +track_half, rear_tire_path, 2),
        ("RearRight", -wheelbase_half, -track_half, rear_tire_path, 3),
    ]
    for name, wx, wy, tire_path, idx in wheel_specs:
        wpath = f"{vehicle_path}/{name}"
        wpos = Gf.Vec3f(float(wx), float(wy), float(wheel_rest_z))
        spos = Gf.Vec3f(float(wx), float(wy), float(susp_frame_z))
        wxform = UsdGeom.Xform.Define(stage, wpath)
        wxform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionFloat).Set(wpos)
        wattAPI = PhysxSchema.PhysxVehicleWheelAttachmentAPI.Apply(wxform.GetPrim())
        wattAPI.CreateWheelRel().AddTarget(wheel_path)
        wattAPI.CreateTireRel().AddTarget(tire_path)
        wattAPI.CreateSuspensionRel().AddTarget(susp_path)
        wattAPI.CreateCollisionGroupRel().AddTarget(grp_query)
        wattAPI.CreateSuspensionTravelDirectionAttr(Gf.Vec3f(0, 0, -1))
        wattAPI.CreateSuspensionFramePositionAttr(spos)
        wattAPI.CreateIndexAttr(idx)

        # Wheel collision cylinder (axis along Y = side)
        collpath = f"{wpath}/Collision"
        coll = UsdGeom.Cylinder.Define(stage, collpath)
        coll.CreatePurposeAttr(UsdGeom.Tokens.guide)
        coll.CreateHeightAttr(wheel_width)
        coll.CreateRadiusAttr(wheel_radius)
        coll.CreateAxisAttr(UsdGeom.Tokens.y)
        coll.CreateExtentAttr(UsdGeom.Cylinder.ComputeExtentFromPlugins(coll, 0))
        UsdPhysics.CollisionAPI.Apply(coll.GetPrim())
        add_collision_to_collision_group(stage, collpath, grp_wheel)
        pcoll = PhysxSchema.PhysxCollisionAPI.Apply(coll.GetPrim())
        pcoll.CreateRestOffsetAttr(0.0)
        pcoll.CreateContactOffsetAttr(0.02)

        # Render cylinder
        rpath = f"{wpath}/Render"
        rcyl = UsdGeom.Cylinder.Define(stage, rpath)
        rcyl.CreateHeightAttr(wheel_width)
        rcyl.CreateRadiusAttr(wheel_radius)
        rcyl.CreateAxisAttr(UsdGeom.Tokens.y)
        rcyl.CreateExtentAttr(UsdGeom.Cylinder.ComputeExtentFromPlugins(rcyl, 0))
        rcyl.CreateDisplayColorAttr([Gf.Vec3f(0.1, 0.1, 0.1)])

    # Chassis collision box
    chassis_path = f"{vehicle_path}/ChassisCollision"
    chass = UsdGeom.Cube.Define(stage, chassis_path)
    chass.CreatePurposeAttr(UsdGeom.Tokens.guide)
    chass.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionFloat).Set(Gf.Vec3f(0, 0, 0))
    chass.AddScaleOp(precision=UsdGeom.XformOp.PrecisionFloat).Set(
        Gf.Vec3f(box[0] * 0.5, box[1] * 0.5, chassis_half_height)
    )
    UsdPhysics.CollisionAPI.Apply(chass.GetPrim())
    add_collision_to_collision_group(stage, chassis_path, grp_chassis)
    pcoll = PhysxSchema.PhysxCollisionAPI.Apply(chass.GetPrim())
    pcoll.CreateRestOffsetAttr(0.0)
    pcoll.CreateContactOffsetAttr(0.02)

    # Chassis render box
    rchass_path = f"{vehicle_path}/ChassisRender"
    rchass = UsdGeom.Cube.Define(stage, rchass_path)
    rchass.AddScaleOp(precision=UsdGeom.XformOp.PrecisionFloat).Set(
        Gf.Vec3f(box[0] * 0.5, box[1] * 0.4, chassis_half_height)
    )
    rchass.CreateDisplayColorAttr([Gf.Vec3f(0.28, 0.65, 1.0)])

    # Bind tarmac material to the ground later (done after ground spawn)
    return veh_prim, ctrl, tarmac_path, grp_ground


def main() -> int:
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1.0 / 60.0, device="cpu")
    )

    import omni.usd
    from pxr import PhysxSchema, UsdGeom, UsdPhysics, Sdf
    from omni.physx.scripts.physicsUtils import (
        add_collision_to_collision_group,
        add_physics_material_to_prim,
    )

    stage = omni.usd.get_context().get_stage()

    # Z-up world (Isaac default). Apply VehicleContext to existing PhysicsScene.
    scene_prim = stage.GetPrimAtPath("/physicsScene")
    if not scene_prim or not scene_prim.IsValid():
        # fallback: create one
        scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
        scene.CreateGravityDirectionAttr((0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr(9.81)
        scene_prim = scene.GetPrim()
    vctx = PhysxSchema.PhysxVehicleContextAPI.Apply(scene_prim)
    vctx.CreateUpdateModeAttr(PhysxSchema.Tokens.velocityChange)
    vctx.CreateVerticalAxisAttr(PhysxSchema.Tokens.posZ)
    vctx.CreateLongitudinalAxisAttr(PhysxSchema.Tokens.posX)

    sim_utils.DomeLightCfg(intensity=2500.0, color=(0.95, 0.95, 0.95)).func(
        "/World/Light",
        sim_utils.DomeLightCfg(intensity=2500.0, color=(0.95, 0.95, 0.95)),
    )

    # Build vehicle
    vehicle_path = "/World/BoxCar"
    veh_prim, ctrl, tarmac_path, grp_ground = build_vehicle(
        stage, vehicle_path, init_xy=(0.0, 0.0)
    )

    # Ground plane (custom, so we can put it in GroundSurface collision group)
    gp_path = "/World/Ground"
    gp = UsdGeom.Mesh.Define(stage, gp_path)
    from pxr import Gf
    gp.CreatePointsAttr(
        [
            Gf.Vec3f(-50, -50, 0),
            Gf.Vec3f(50, -50, 0),
            Gf.Vec3f(50, 50, 0),
            Gf.Vec3f(-50, 50, 0),
        ]
    )
    gp.CreateFaceVertexCountsAttr([4])
    gp.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    gp.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * 4)
    gp.CreateDisplayColorAttr([Gf.Vec3f(0.5, 0.5, 0.5)])

    coll_path = f"{gp_path}/CollisionPlane"
    coll = UsdGeom.Plane.Define(stage, coll_path)
    coll.CreatePurposeAttr(UsdGeom.Tokens.guide)
    coll.CreateAxisAttr(UsdGeom.Tokens.z)
    UsdPhysics.CollisionAPI.Apply(coll.GetPrim())
    add_collision_to_collision_group(stage, coll_path, grp_ground)
    add_physics_material_to_prim(stage, coll.GetPrim(), Sdf.Path(tarmac_path))

    chase = ChaseCamRecorder(
        out_dir=OUT_DIR,
        tag=f"physx_vehicle_{args.label}",
        prim_path="/World/ChaseCam",
        behind=6.0, above=2.5, look_up=0.5,
        width=960, height=540,
    )

    sim.reset()
    chase.on_reset()

    dt = sim.get_physics_dt()
    frame_step = max(1, int(round(1.0 / (args.fps * dt))))
    step_times = []

    # Helper: read vehicle world pose via PhysX rigid body interface
    from omni.physx import get_physx_interface
    physx_iface = get_physx_interface()

    accel_attr = ctrl.GetAcceleratorAttr()
    steer_attr = ctrl.GetSteerAttr()

    for i in range(args.steps):
        t = i * dt
        # Ramp accelerator over first 1 s, then hold
        a = min(args.accel, args.accel * (t / 1.0))
        accel_attr.Set(float(a))
        steer_attr.Set(float(args.steer_amp * math.sin(2 * math.pi * args.steer_freq * t)))

        t0 = time.perf_counter()
        sim.step()
        step_times.append(time.perf_counter() - t0)

        if i % frame_step == 0:
            # Read pose from USD xform (PhysX writes it back each step)
            xform_cache = UsdGeom.XformCache()
            mat = xform_cache.GetLocalToWorldTransform(veh_prim)
            tr = mat.ExtractTranslation()
            rot = mat.ExtractRotationQuat()
            # Quat -> yaw (Z-up)
            qw = rot.GetReal()
            im = rot.GetImaginary()
            qx, qy, qz = im[0], im[1], im[2]
            yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
            chase.update_follow((float(tr[0]), float(tr[1]), float(tr[2])), yaw)
            chase.capture_frame()

            if i % (frame_step * 10) == 0:
                print(
                    f"[veh] step={i} t={t:.2f}s pos=({tr[0]:.2f},{tr[1]:.2f},{tr[2]:.2f}) "
                    f"yaw={math.degrees(yaw):+.1f}deg accel={a:.2f}",
                    flush=True,
                )

    out = chase.finalize()
    print(f"PHYSX_VEH_DONE outputs={out}", flush=True)

    import statistics
    warm = step_times[30:]
    if warm:
        print(
            f"STEP_STATS label={args.label} "
            f"median={statistics.median(warm)*1000:.3f}ms "
            f"mean={statistics.mean(warm)*1000:.3f}ms "
            f"p95={sorted(warm)[int(len(warm)*0.95)]*1000:.3f}ms",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    import traceback
    rc = 0
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    os._exit(rc)
