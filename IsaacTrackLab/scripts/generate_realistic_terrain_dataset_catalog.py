#!/usr/bin/env python3
"""Generate 60 semantically planned city, park, and forest maps."""

from __future__ import annotations

import json
from math import atan2, degrees, hypot
from pathlib import Path
import random


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "configs/terrain_dataset_v3_realistic.json"


def distance_to_segment(px, py, ax, ay, bx, by):
    dx, dy = bx-ax, by-ay
    denominator = dx*dx + dy*dy
    if denominator <= 1e-9:
        return hypot(px-ax, py-ay)
    ratio = min(1.0, max(0.0, ((px-ax)*dx+(py-ay)*dy)/denominator))
    return hypot(px-(ax+ratio*dx), py-(ay+ratio*dy))


def distance_to_corridors(x, y, corridors):
    return min(
        distance_to_segment(x, y, *left, *right)
        for corridor in corridors
        for left, right in zip(corridor["points_m"], corridor["points_m"][1:])
    )


def gate(point, size=6.0):
    return [round(point[0], 3), round(point[1], 3), size, size]


def midpoint(left, right):
    return [(float(left[0]) + float(right[0])) * 0.5,
            (float(left[1]) + float(right[1])) * 0.5]


def lerp(left, right, ratio):
    return [
        float(left[0]) + ratio * (float(right[0]) - float(left[0])),
        float(left[1]) + ratio * (float(right[1]) - float(left[1])),
    ]


def travel_yaw(left, right):
    return degrees(atan2(float(right[1]) - float(left[1]), float(right[0]) - float(left[0])))


def gait_feature(feature_id, kind, center, yaw_deg, **parameters):
    """Create a route-mandatory, semantically labelled locomotion feature."""

    return {
        "id": feature_id, "kind": kind,
        "center_m": [round(float(center[0]), 3), round(float(center[1]), 3)],
        "yaw_deg": round(float(yaw_deg), 2),
        "required_on_route": True,
        **parameters,
    }


def instance(asset, x, y, scale, proxy, yaw=0.0, **shape):
    return {
        "id": "", "asset": asset, "x_m": round(x, 3), "y_m": round(y, 3),
        "yaw_deg": round(yaw, 2), "scale": round(scale, 4), "proxy": proxy, **shape,
    }


def assign_ids(items):
    counts = {}
    for item in items:
        counts[item["asset"]] = counts.get(item["asset"], 0) + 1
        item["id"] = f"{item['asset']}_{counts[item['asset']]:03d}"
    return items


def sample_off_corridor(rng, corridors, count, minimum_distance, zones=(), occupied=None):
    occupied = occupied if occupied is not None else []
    output = []
    for _ in range(count):
        for _ in range(4000):
            x, y = rng.uniform(-82, 82), rng.uniform(-82, 82)
            if distance_to_corridors(x, y, corridors) < minimum_distance:
                continue
            if any(abs(x-z[0]) < z[2]*0.62 and abs(y-z[1]) < z[3]*0.62 for z in zones):
                continue
            if any(hypot(x-ox, y-oy) < minimum_distance*0.72 for ox,oy in occupied):
                continue
            occupied.append((x,y)); output.append((x,y)); break
        else:
            raise RuntimeError("could not sample semantic prop placement")
    return output


def city_map(index, seed):
    rng = random.Random(seed)
    shift_x, shift_y = rng.uniform(-4,4), rng.uniform(-4,4)
    xs = [-55+shift_x, 4+shift_x, 59+shift_x]
    ys = [-58+shift_y, 1+shift_y, 57+shift_y]
    road_width = rng.uniform(10.0, 12.0)
    roads = []
    for i, y in enumerate(ys):
        roads.append({"id": f"road_h_{i}", "points_m": [[-88,y],[88,y]], "width_m": road_width})
    for i, x in enumerate(xs):
        roads.append({"id": f"road_v_{i}", "points_m": [[x,-88],[x,88]], "width_m": road_width})
    surfaces = []
    for road in roads:
        surfaces.append({**road, "id": f"sidewalk_{road['id']}", "width_m": road_width+4.4,
                         "texture": "brick_pavement_02", "kind": "sidewalk", "z_offset_m": 0.0})
        surfaces.append({**road, "texture": "asphalt_02", "kind": "road", "z_offset_m": 0.0})
    nodes = {
        "sw": (-78,ys[0]), "se": (78,ys[0]), "mw": (-78,ys[1]), "me": (78,ys[1]),
        "nw": (-78,ys[2]), "ne": (78,ys[2]),
        "a": (xs[0],ys[0]), "b": (xs[1],ys[0]), "c": (xs[2],ys[0]),
        "d": (xs[0],ys[1]), "e": (xs[1],ys[1]), "f": (xs[2],ys[1]),
        "g": (xs[0],ys[2]), "h": (xs[1],ys[2]), "i": (xs[2],ys[2]),
    }
    patterns = [
        ("sw","c","f","d"), ("se","a","d","i"), ("nw","i","f","b"),
        ("ne","g","d","c"), ("sw","h","i","me"),
    ]
    gate_sets = [[gate(nodes[name]) for name in pattern] for pattern in patterns]
    bounds = [-88, *xs, 88]
    ybounds = [-88, *ys, 88]
    structures = []
    for xi in range(4):
        for yi in range(4):
            left, right = bounds[xi], bounds[xi+1]
            bottom, top = ybounds[yi], ybounds[yi+1]
            if right-left < 18 or top-bottom < 18:
                continue
            width = min(rng.uniform(10,17), right-left-road_width-5)
            depth = min(rng.uniform(9,16), top-bottom-road_width-5)
            if width < 7 or depth < 7:
                continue
            structures.append({
                "id": f"building_{len(structures):02d}", "kind": "building",
                "center_m": [round((left+right)/2+rng.uniform(-2,2),3), round((bottom+top)/2+rng.uniform(-2,2),3)],
                "size_m": [round(width,3),round(depth,3),round(rng.uniform(8,22),3)],
                "material": rng.choice(("brick_wall_001","beige_wall_001")), "yaw_deg": 0.0,
            })
    props = []
    roadside_occupied = []
    def roadside_clear(x, y, radius, margin=.55):
        return all(hypot(x-ox, y-oy) >= radius+other_radius+margin
                   for ox,oy,other_radius in roadside_occupied)
    # Parked cars and street furniture are aligned to curbs, never scattered in blocks.
    for n in range(rng.randint(10,14)):
        for _ in range(1000):
            horizontal = rng.random() < 0.55
            scale = rng.uniform(.88,1.06)
            if horizontal:
                y = rng.choice(ys) + rng.choice((-1,1))*road_width*0.32
                x = rng.uniform(-78,78); yaw = rng.choice((0.0,180.0))
                if any(abs(x-intersection_x) < 10.0 for intersection_x in xs): continue
            else:
                x = rng.choice(xs) + rng.choice((-1,1))*road_width*0.32
                y = rng.uniform(-78,78); yaw = rng.choice((90.0,-90.0))
                if any(abs(y-intersection_y) < 10.0 for intersection_y in ys): continue
            radius = .5*hypot(2.1,4.7)*scale
            if roadside_clear(x,y,radius,3.0): break
        else: raise RuntimeError("could not place separated parked cars")
        roadside_occupied.append((x,y,radius))
        props.append(instance("covered_car",x,y,scale,"box",yaw,size_m=[4.7,2.1,1.6]))
    for n in range(rng.randint(12,18)):
        for _ in range(1000):
            horizontal = rng.random() < .5
            scale = rng.uniform(.85,1.1); radius=.42*scale
            if horizontal:
                x,y=rng.uniform(-82,82),rng.choice(ys)+rng.choice((-1,1))*(road_width/2+1.25)
            else:
                x,y=rng.choice(xs)+rng.choice((-1,1))*(road_width/2+1.25),rng.uniform(-82,82)
            if roadside_clear(x,y,radius): break
        else: raise RuntimeError("could not place separated street furniture")
        roadside_occupied.append((x,y,radius))
        props.append(instance(rng.choice(("metal_trash_can","fire_hydrant")),x,y,scale,"cylinder",
                              radius_m=.42,height_m=1.0))
    curb_segments=[]
    intervals=lambda lines: [(-86,lines[0]-8),*( (lines[i]+8,lines[i+1]-8) for i in range(len(lines)-1) ),(lines[-1]+8,86)]
    for y in ys:
        for left,right in intervals(xs):
            if right-left>2:
                for side in (-1,1): curb_segments.append({"center_m":[(left+right)/2,y+side*road_width/2],"length_m":right-left,"yaw_deg":0.0})
    for x in xs:
        for bottom,top in intervals(ys):
            if top-bottom>2:
                for side in (-1,1): curb_segments.append({"center_m":[x+side*road_width/2,(bottom+top)/2],"length_m":top-bottom,"yaw_deg":90.0})
    terrain={
        "profile":"rugged_city","texture":"brick_pavement_02","hdri":"urban_street_01","seed":seed,
        "amplitude_m":round(rng.uniform(1.72,2.30),4),"frequency":round(rng.uniform(.028,.038),5),
        "roughness_m":round(rng.uniform(.20,.34),4),"micro_frequency":round(rng.uniform(.075,.11),5),
        "tilt_x":round(rng.uniform(-.005,.005),5),"tilt_y":round(rng.uniform(-.005,.005),5),
        "static_friction":.94,"dynamic_friction":.80,
        "features":[
            {"id":"gentle_grade","kind":"gaussian","center_m":[rng.uniform(-40,40),rng.uniform(-40,40)],"amplitude_m":rng.uniform(.35,.65),"sigma_m":rng.uniform(24,34)},
        ],
    }
    zones=[{"id":"road_repair","kind":"rough_ground","center_m":[xs[1]+18,ys[1]],"size_m":[18,road_width*.8],"cost":1.7,"traversable":True,"static_friction":.70,"dynamic_friction":.52}]
    gait_features = []
    if index == 1:
        course_left, course_right = nodes["d"], nodes["e"]
        course_yaw = travel_yaw(course_left, course_right)
        gait_features = [
            gait_feature(
                "urban_raised_curb_crossing", "curb_table",
                lerp(course_left, course_right, 0.35), course_yaw,
                width_m=2.8, length_m=1.2, height_m=0.10,
                material="curb", gait_hint="step_up_down",
            ),
            gait_feature(
                "urban_service_platform", "platform",
                lerp(course_left, course_right, 0.50), course_yaw,
                width_m=2.8, length_m=2.8, height_m=0.18,
                material="curb", gait_hint="low_jump_up_down",
            ),
            gait_feature(
                "urban_plaza_stairs", "stair_platform",
                lerp(course_left, course_right, 0.65), course_yaw,
                width_m=3.0, step_count=3, step_rise_m=0.08,
                tread_depth_m=0.46, platform_length_m=1.8,
                material="curb", gait_hint="stair_transition",
            ),
        ]
    return base_map(
        "city", index, seed, terrain, roads, surfaces, gate_sets, zones,
        assign_ids(props), structures, curb_segments, gait_features,
    )


def park_map(index, seed):
    rng=random.Random(seed)
    lake=(rng.choice((-28,28))+rng.uniform(-4,4),rng.choice((-18,20))+rng.uniform(-4,4))
    basin_rx,basin_ry=rng.uniform(18,24),rng.uniform(13,18)
    loop=[(-72,-38),(-62,20),(-35,62),(8,75),(52,58),(75,18),(67,-38),(35,-70),(-18,-76),(-58,-62),(-72,-38)]
    loop=[[x+rng.uniform(-3,3),y+rng.uniform(-3,3)] for x,y in loop]
    west_east=[[-75,-4],[-42,-12],[-8,-5],[28,5],[72,4]]
    north_south=[[-12,-76],[-18,-38],[-8,0],[-15,38],[-8,74]]
    corridors=[
        {"id":"park_loop","points_m":loop,"width_m":6.0},
        {"id":"park_cross_e","points_m":west_east,"width_m":5.5},
        {"id":"park_cross_n","points_m":north_south,"width_m":5.5},
    ]
    surfaces=[{**c,"texture":"rock_path","kind":"park_path","z_offset_m":0.0} for c in corridors]
    # Use ordered points on the continuous loop as route gates.  The former
    # catalogue used the tips of the two cross paths as gates; after reaching a
    # tip, A* had no option except an instantaneous 180-degree retrace.  The
    # cross paths remain available as valid shortcuts, but are no longer used
    # as dead-end task goals.
    loop_nodes=loop[:-1]
    patterns=(
        tuple((start+2*step) % len(loop_nodes) for step in range(4))
        for start in range(len(loop_nodes))
    )
    reverse_patterns=(
        tuple((start-2*step) % len(loop_nodes) for step in range(4))
        for start in range(len(loop_nodes))
    )
    gate_sets=[
        [gate(loop_nodes[node_index],5.2) for node_index in pattern]
        for pattern in (*patterns,*reverse_patterns)
    ]
    lake_zone={"id":"lake","kind":"water","shape":"ellipse","center_m":[round(lake[0],3),round(lake[1],3)],
               "size_m":[round(basin_rx*1.12,3),round(basin_ry*1.12,3)],"cost":99.0,"traversable":False,"water_raise_m":.76}
    terrain={
        "profile":"rolling_park","texture":"leafy_grass","hdri":"je_gray_park","seed":seed,
        "amplitude_m":round(rng.uniform(2.62,3.45),4),"frequency":round(rng.uniform(.025,.034),5),
        "roughness_m":round(rng.uniform(.22,.36),4),"micro_frequency":round(rng.uniform(.075,.11),5),
        "tilt_x":round(rng.uniform(-.006,.006),5),"tilt_y":round(rng.uniform(-.006,.006),5),
        "static_friction":.86,"dynamic_friction":.68,
        "features":[
            {"id":"lake_basin","kind":"lake_basin","center_m":list(lake),"amplitude_m":1.45,"radius_x_m":basin_rx,"radius_y_m":basin_ry},
            {"id":"hill_west","kind":"gaussian","center_m":[-48,35],"amplitude_m":rng.uniform(1.2,2.0),"sigma_m":rng.uniform(18,25)},
            {"id":"hill_south","kind":"gaussian","center_m":[28,-48],"amplitude_m":rng.uniform(.9,1.7),"sigma_m":rng.uniform(17,24)},
        ],
    }
    occupied=[]; zones=[(lake[0],lake[1],basin_rx*2,basin_ry*2)]
    props=[]
    for x,y in sample_off_corridor(rng,corridors,rng.randint(48,62),5.2,zones,occupied):
        props.append(instance("pine_tree_01",x,y,rng.uniform(.68,1.05),"cylinder",radius_m=.5,height_m=6.5))
    for x,y in sample_off_corridor(rng,corridors,rng.randint(12,18),4.2,zones,occupied):
        props.append(instance("boulder_01",x,y,rng.uniform(.42,.82),"sphere",radius_m=.9))
    for n,p in enumerate(loop[1:9:2]):
        q=loop[(1+2*n+1)%len(loop)]; yaw=degrees(atan2(q[1]-p[1],q[0]-p[0]))
        props.append(instance("painted_wooden_bench",p[0]+3.8,p[1]+3.0,rng.uniform(.9,1.05),"box",yaw,size_m=[1.9,.75,1.2]))
    step_features=[]
    for n,p in enumerate((loop[1],loop[5],west_east[2])):
        step_features.append({"id":f"path_stones_{n}","kind":"embedded_stones","center_m":[round(p[0],3),round(p[1],3)],"yaw_deg":rng.uniform(-180,180),"count":3})
        terrain["features"].append({"id":f"path_bump_{n}","kind":"step_bump","center_m":p,"amplitude_m":.10,"radius_x_m":1.6,"radius_y_m":.9,"angle_deg":rng.uniform(-180,180)})
    if index == 1:
        course_left, course_right = loop[0], loop[1]
        course_yaw = travel_yaw(course_left, course_right)
        step_features.extend((
            gait_feature(
                "park_path_curb", "curb_table", lerp(course_left, course_right, 0.35),
                course_yaw, width_m=2.5, length_m=1.0,
                height_m=0.10, material="rock", gait_hint="step_up_down",
            ),
            gait_feature(
                "park_timber_platform", "platform", lerp(course_left, course_right, 0.50),
                course_yaw, width_m=2.6, length_m=3.0,
                height_m=0.16, material="wood", gait_hint="low_jump_up_down",
            ),
            gait_feature(
                "park_terrace_stairs", "stair_platform", lerp(course_left, course_right, 0.65),
                course_yaw, width_m=2.7, step_count=3,
                step_rise_m=0.08, tread_depth_m=0.48, platform_length_m=1.8,
                material="rock", gait_hint="stair_transition",
            ),
        ))
    return base_map("park",index,seed,terrain,corridors,surfaces,gate_sets,[lake_zone],assign_ids(props),[],[],step_features)


def forest_map(index, seed):
    rng=random.Random(seed)
    main=[[-82,-67],[-58,-42],[-72,-8],[-46,26],[-60,61],[-18,74],[18,57],[2,22],[37,-5],[18,-38],[65,-68],[84,-48]]
    main=[[x+rng.uniform(-3,3),y+rng.uniform(-3,3)] for x,y in main]
    branch1=[main[2],[-20,-4],[22,18],main[6]]
    branch2=[main[4],[-5,36],[38,42],[70,20]]
    corridors=[
        {"id":"main_mountain_trail","points_m":main,"width_m":6.2},
        {"id":"forest_branch_a","points_m":branch1,"width_m":5.5},
        {"id":"forest_branch_b","points_m":branch2,"width_m":5.2},
    ]
    surfaces=[{**c,"texture":"muddy_tracks" if n==0 else "rocky_trail","kind":"forest_trail","z_offset_m":0.0} for n,c in enumerate(corridors)]
    base_patterns=(
        (0,3,6,9),(1,4,7,10),(0,2,5,8,10),(4,6,8,11),(10,7,4,1),
        (2,5,7,10),(0,4,6,9),(1,3,6,10),(11,8,5,2),(9,6,4,0),
    )
    patterns=base_patterns+tuple(tuple(reversed(pattern)) for pattern in base_patterns)
    gate_sets=[[gate(main[i],5.4) for i in pattern] for pattern in patterns]
    terrain={
        "profile":"forest_ridge","texture":"forest_ground_04","hdri":"forest_grove","seed":seed,
        "amplitude_m":round(rng.uniform(3.8,5.4),4),"frequency":round(rng.uniform(.026,.036),5),
        "roughness_m":round(rng.uniform(.48,.78),4),"micro_frequency":round(rng.uniform(.09,.13),5),
        "tilt_x":round(rng.uniform(-.012,.012),5),"tilt_y":round(rng.uniform(-.012,.012),5),
        "static_friction":.88,"dynamic_friction":.68,
        "features":[
            {"id":"ridge","kind":"ridge","center_m":[rng.uniform(-30,30),rng.uniform(-30,30)],"amplitude_m":rng.uniform(1.2,2.0),"sigma_m":rng.uniform(13,19),"angle_deg":rng.uniform(-180,180),"wavelength_m":18},
            {"id":"ravine","kind":"rut","center_m":[rng.uniform(-25,25),rng.uniform(-25,25)],"amplitude_m":rng.uniform(.8,1.4),"sigma_m":rng.uniform(10,15),"angle_deg":rng.uniform(-180,180),"wavelength_m":16},
        ],
    }
    occupied=[]; props=[]
    for x,y in sample_off_corridor(rng,corridors,rng.randint(92,118),4.8,(),occupied):
        props.append(instance("pine_tree_01",x,y,rng.uniform(.72,1.2),"cylinder",radius_m=.52,height_m=7.0))
    for x,y in sample_off_corridor(rng,corridors,rng.randint(22,30),4.0,(),occupied):
        props.append(instance(rng.choice(("boulder_01","rock_09")),x,y,rng.uniform(.48,1.0),"sphere",radius_m=.95))
    for x,y in sample_off_corridor(rng,corridors,rng.randint(14,20),6.2,(),occupied):
        props.append(instance(rng.choice(("dead_tree_trunk","tree_stump_01")),x,y,rng.uniform(.72,1.15),
                              "box",rng.uniform(-180,180),size_m=[4.2,.9,.9]))
    step_features=[]
    for n,point_index in enumerate((1,3,5,7,9)):
        p=main[point_index]; prev=main[point_index-1]; nxt=main[point_index+1]
        trail_yaw=degrees(atan2(nxt[1]-prev[1],nxt[0]-prev[0])); yaw=trail_yaw+90
        kind="low_log" if n%2==0 else "embedded_stones"
        step_features.append({"id":f"trail_step_{n}","kind":kind,"center_m":[round(p[0],3),round(p[1],3)],"yaw_deg":round(yaw,2),"count":3})
        terrain["features"].append({"id":f"trail_step_bump_{n}","kind":"step_bump","center_m":p,
                                    "amplitude_m":round(rng.uniform(.10,.17),3),"radius_x_m":1.7,"radius_y_m":.75,"angle_deg":yaw})
    if index == 1:
        course_left, course_right = main[5], main[6]
        course_yaw = travel_yaw(course_left, course_right)
        step_features.extend((
            gait_feature(
                "forest_natural_ledge", "curb_table", lerp(course_left, course_right, 0.30),
                course_yaw, width_m=2.6, length_m=1.1,
                height_m=0.12, material="rock", gait_hint="step_up_down",
            ),
            gait_feature(
                "forest_boardwalk_platform", "platform", lerp(course_left, course_right, 0.50),
                course_yaw, width_m=2.7, length_m=3.2,
                height_m=0.18, material="wood", gait_hint="low_jump_up_down",
            ),
            gait_feature(
                "forest_rock_stairs", "stair_platform", lerp(course_left, course_right, 0.70),
                course_yaw, width_m=2.8, step_count=3,
                step_rise_m=0.09, tread_depth_m=0.50, platform_length_m=1.7,
                material="rock", gait_hint="stair_transition",
            ),
        ))
    zones=[{"id":"mud_section","kind":"soft_ground","center_m":main[7],"size_m":[12,8],"cost":1.8,"traversable":True,"static_friction":.56,"dynamic_friction":.38}]
    return base_map("forest_trail",index,seed,terrain,corridors,surfaces,gate_sets,zones,assign_ids(props),[],[],step_features)


def base_map(family,index,seed,terrain,corridors,surfaces,gate_sets,zones,props,structures,curbs,steps):
    labels={"city":"真实城市道路","park":"湖泊丘陵公园","forest_trail":"崎岖森林泥土山径"}
    target_slope={"city":14.0,"park":17.0,"forest_trail":19.0}
    return {
        "id":f"{family}_{index:02d}","family":family,"label":f"{labels[family]} {index:02d}","seed":seed,
        "bounds_m":[-90.0,90.0,-90.0,90.0],"target_maximum_slope_deg":target_slope[family],
        "terrain":terrain,"navigation_corridors":corridors,"surface_corridors":surfaces,
        "route_gates":gate_sets[0],"route_gate_sets":gate_sets,"challenge_zones":zones,
        "obstacle_instances":props,"obstacle_templates":[],"procedural_structures":structures,
        "curb_segments":curbs,"traversable_step_features":steps,
    }


def main():
    maps=[]
    for index in range(1,21): maps.append(city_map(index,171000+index*101))
    for index in range(1,21): maps.append(park_map(index,182000+index*101))
    for index in range(1,21): maps.append(forest_map(index,193000+index*101))
    catalog={
        "schema_version":3,"batch_id":"city_park_forest_realistic_v3",
        "scene_output_root":"assets/scenes/terrain_dataset_v3_realistic","maps_per_family":20,"routes_per_map":20,
        "quality_contract":{"texture_resolution":"2k","terrain_grid_spacing_m":1.0,"terrain_buffer_m":30.0,"usd_extension":"usdc","collision_required":True,"target_routes_must_be_reachable":True,"dynamic_events":[],"distractor_people":0},
        "planning_contract":{
            "semantic_layout_required":True,"routes_must_follow_navigation_corridors":True,
            "required_route_features_must_be_visited":True,
            "city_requires":["road_network","sidewalks","curb_breaks_at_intersections","roadside_props","zoned_buildings"],
            "park_requires":["terrain_bowl_lake","elliptical_water_surface","hills","connected_paths","shore_and_path_stones"],
            "forest_requires":["dense_off-trail_trees","rugged_ridges","mud_trails","rocks","deadwood","traversable_step_features"],
        },
        "route_defaults":{"algorithm":"semantic_corridor_cost_aware_astar","grid_resolution_m":1.0,"resample_spacing_m":1.0,"robot_clearance_m":1.2,"maximum_slope_deg":28.0,"target_maximum_slope_deg":18.0,"maximum_turn_deg":150.0,"maximum_traversable_step_height_m":0.30,"minimum_length_m":280.0,"maximum_length_m":360.0,"maximum_attempts":120},
        "maps":maps,
    }
    OUTPUT.write_text(json.dumps(catalog,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("REALISTIC_TERRAIN_CATALOG_OK",len(maps),OUTPUT)


if __name__ == "__main__": main()
