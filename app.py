from __future__ import annotations

import html
import io
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
from shapely.geometry import LineString, Point, Polygon
from streamlit_folium import st_folium

try:
    import scipy.optimize as opt

    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False


APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
QATAR_CRS = "EPSG:4326"
QATAR_METRIC_CRS = "EPSG:2933"
ELECTRICITY_QAR_KWH = 0.20
WATER_QAR_M3 = 5.0
COOLING_HOURS_YEAR = 1900
T_SET_C = 25.0
CGIS_GROUNDWATER_WELLS_URL = (
    "https://services.gisqatar.org.qa/server/rest/services/Vector/Water/MapServer/1/query"
    "?where=SUBTYPECD%3D2"
    "&outFields=OBJECTID,SUBTYPEDESCRIPTION,LIFECYCLESTATUS,ASSETID,DATASOURCE"
    "&returnGeometry=true&f=geojson&outSR=4326"
)
ENABLE_REMOTE_GIS = os.environ.get("GREENHOUSE_ATLAS_REMOTE_GIS", "0") == "1"


QATAR_POLYGON = Polygon(
    [
        (50.755, 24.595),
        (50.925, 24.570),
        (51.130, 24.590),
        (51.355, 24.705),
        (51.520, 24.905),
        (51.595, 25.090),
        (51.635, 25.300),
        (51.590, 25.545),
        (51.515, 25.760),
        (51.405, 25.965),
        (51.245, 26.135),
        (51.065, 26.155),
        (50.925, 25.980),
        (50.805, 25.720),
        (50.755, 25.390),
        (50.745, 25.020),
        (50.755, 24.595),
    ]
)


REFERENCE_STATIONS = {
    "Doha": (51.531, 25.285, 41.5, 55.0, 890.0),
    "Al Khor": (51.507, 25.684, 39.8, 62.0, 870.0),
    "Al Shahaniya": (51.205, 25.371, 43.2, 48.0, 920.0),
    "Mesaieed": (51.553, 24.998, 42.8, 52.0, 905.0),
    "Ruwais": (51.221, 26.142, 39.2, 65.0, 860.0),
}

NATIONAL_GW_SAFE_YIELD_M3_YEAR = 57_200_000.0
NATIONAL_GW_ABSTRACTION_M3_YEAR = 250_000_000.0
NATIONAL_GW_STRESS_RATIO = NATIONAL_GW_ABSTRACTION_M3_YEAR / NATIONAL_GW_SAFE_YIELD_M3_YEAR

GW_SOURCE_SUMMARY = (
    "Composite groundwater score uses public/peer-reviewed screening defaults: CGIS Qatar for hydrogeological "
    "station geometry; Ajjur & Al-Ghamdi 2022 for Kahramaa 2021 groundwater-level map with 313 monitored wells "
    "and northern/southern basin context; Bilal et al. 2025 for salinity, transmissivity, and storativity classes "
    "sourced to Ministry of Environment / Schlumberger Water Services 2009; USGS OFR 85-343 for recharge context. "
    "Exact official well quality, yield, level, and permit fields override these proxies when present."
)

GW_REQUIRED_FIELDS = [
    "tds_mg_l",
    "water_level_m_bgl",
    "well_yield_m3_day",
    "permitted_abstraction_m3_year",
    "transmissivity_m2_day",
    "storativity",
    "aquifer_stress_ratio",
]


@dataclass(frozen=True)
class CropProfile:
    name: str
    crop_family: str
    yield_kg_m2_year: float
    kc: float
    price_qar_kg: float
    water_sensitivity: float
    climate_sensitivity: float
    preferred_zone: str
    notes: str


@dataclass(frozen=True)
class GreenhouseTech:
    name: str
    category: str
    capital_qar_m2: float
    fixed_opex_qar_m2_year: float
    cooling_mode: str
    cop: float
    pad_efficiency: float
    lighting_w_m2: float
    yield_multiplier: float
    water_multiplier: float
    energy_multiplier: float
    labour_factor: float
    description: str


@dataclass(frozen=True)
class GroundwaterRegionalProfile:
    zone: str
    source_tier: str
    source_note: str
    tds_mg_l: float
    water_level_m_bgl: float
    well_yield_m3_day: float
    permitted_abstraction_m3_year: float
    transmissivity_m2_day: float
    storativity: float
    aquifer_stress_ratio: float
    confidence: float


GROUNDWATER_REGIONAL_PROFILES: Dict[str, GroundwaterRegionalProfile] = {
    "Northern aquifer / farm belt": GroundwaterRegionalProfile(
        zone="Northern aquifer / farm belt",
        source_tier="B/C proxy",
        source_note="Ajjur & Al-Ghamdi 2022; Bilal et al. 2025; Schlumberger/MoE 2009 cited classes",
        tds_mg_l=2_200.0,
        water_level_m_bgl=30.0,
        well_yield_m3_day=420.0,
        permitted_abstraction_m3_year=np.nan,
        transmissivity_m2_day=500.0,
        storativity=1e-2,
        aquifer_stress_ratio=3.4,
        confidence=0.68,
    ),
    "Central Qatar / transition aquifer": GroundwaterRegionalProfile(
        zone="Central Qatar / transition aquifer",
        source_tier="C proxy",
        source_note="Interpolated from Qatar aquifer literature and Ministry/Schlumberger class ranges",
        tds_mg_l=4_200.0,
        water_level_m_bgl=38.0,
        well_yield_m3_day=250.0,
        permitted_abstraction_m3_year=np.nan,
        transmissivity_m2_day=180.0,
        storativity=1e-3,
        aquifer_stress_ratio=NATIONAL_GW_STRESS_RATIO,
        confidence=0.55,
    ),
    "Southern / southwestern stressed aquifer": GroundwaterRegionalProfile(
        zone="Southern / southwestern stressed aquifer",
        source_tier="C proxy",
        source_note="Ajjur & Al-Ghamdi 2022 reports lower quality than northern aquifer; Bilal et al. 2025 salinity classes",
        tds_mg_l=7_500.0,
        water_level_m_bgl=49.0,
        well_yield_m3_day=120.0,
        permitted_abstraction_m3_year=np.nan,
        transmissivity_m2_day=55.0,
        storativity=1e-4,
        aquifer_stress_ratio=5.2,
        confidence=0.48,
    ),
    "Coastal saline / industrial fringe": GroundwaterRegionalProfile(
        zone="Coastal saline / industrial fringe",
        source_tier="C proxy",
        source_note="Qatar coastal salinity and seawater-intrusion literature; Al-Maktoumi et al. 2025",
        tds_mg_l=9_000.0,
        water_level_m_bgl=24.0,
        well_yield_m3_day=160.0,
        permitted_abstraction_m3_year=np.nan,
        transmissivity_m2_day=110.0,
        storativity=5e-4,
        aquifer_stress_ratio=4.8,
        confidence=0.48,
    ),
}


class AdvancedGreenhouseEngine:
    CP_AIR = 1006.0
    RHO_AIR = 1.204
    LAMBDA_LATENT = 2.45e6
    G_ACCEL = 9.81

    def __init__(self, specs: dict):
        self.length = specs["length"]
        self.width = specs["width"]
        self.height_eaves = specs["height_eaves"]
        self.height_roof = specs["height_roof"]
        self.area_floor = self.length * self.width
        self.volume = self.area_floor * ((self.height_eaves + self.height_roof) / 2.0)
        self.area_cover = specs["area_cover"]
        self.area_vent_roof = specs["area_vent_roof"]
        self.area_vent_side = specs["area_vent_side"]
        self.vent_height_diff = self.height_roof - self.height_eaves
        self.tau_cover = specs["tau_cover"]
        self.net_porosity = specs["net_porosity"]
        self.lai = specs["lai"]
        self.rc_min = specs["rc_min"]

    @staticmethod
    def saturation_vapor_pressure(temp_c: float) -> float:
        return 0.61078 * np.exp((17.27 * temp_c) / (temp_c + 237.3))

    def ventilation_rate(self, temp_in: float, temp_out: float, wind_m_s: float) -> float:
        cd_net = 0.6 * (self.net_porosity**2)
        area_eff = (self.area_vent_roof * self.area_vent_side) / math.sqrt(self.area_vent_roof**2 + self.area_vent_side**2 + 1e-6)
        wind_flow = cd_net * area_eff * max(wind_m_s, 0.1) * math.sqrt(0.22)
        temp_avg_k = (temp_in + temp_out + 273.15) / 2.0
        delta_t = max(abs(temp_in - temp_out), 0.001)
        buoyancy_flow = cd_net * self.area_vent_roof * math.sqrt((2.0 * self.G_ACCEL * self.vent_height_diff * delta_t) / temp_avg_k)
        return math.sqrt(wind_flow**2 + buoyancy_flow**2)

    def evaporative_pad_outlet(self, temp_out: float, rh_out: float, pad_efficiency: float) -> tuple[float, float]:
        # Stull-style wet-bulb approximation keeps the model stable for dashboard use.
        twb = (
            temp_out * math.atan(0.151977 * math.sqrt(rh_out + 8.313659))
            + math.atan(temp_out + rh_out)
            - math.atan(rh_out - 1.676331)
            + 0.00391838 * rh_out**1.5 * math.atan(0.023101 * rh_out)
            - 4.686035
        )
        temp_pad = temp_out - pad_efficiency * (temp_out - twb)
        rh_pad = min(100.0, rh_out + pad_efficiency * (100.0 - rh_out))
        return temp_pad, rh_pad

    def equilibrium_state(self, boundary: dict) -> dict:
        solar = boundary["solar_w_m2"]
        temp_out = boundary["temp_c"]
        rh_out = boundary["rh_pct"]
        wind = boundary.get("wind_m_s", 3.0)
        pad_active = boundary.get("pad_active", False)
        pad_eff = boundary.get("pad_efficiency", 0.84)
        shading = boundary.get("shading_factor", 0.35)

        temp_inlet, rh_inlet = self.evaporative_pad_outlet(temp_out, rh_out, pad_eff) if pad_active else (temp_out, rh_out)
        solar_net = solar * self.tau_cover * (1.0 - shading) * self.area_floor

        def residual(states):
            temp_in, rh_in = float(states[0]), float(np.clip(states[1], 5.0, 100.0))
            vent = self.ventilation_rate(temp_in, temp_out, wind)
            conduction = 5.8 * self.area_cover * (temp_in - temp_out)
            sensible_vent = vent * self.RHO_AIR * self.CP_AIR * (temp_in - temp_inlet)
            p_sat = self.saturation_vapor_pressure(temp_in)
            vapor_pressure = p_sat * rh_in / 100.0
            vpd = max(0.0, p_sat - vapor_pressure)
            aerodynamic_resistance = 200.0 / math.sqrt(max(wind, 0.1))
            stomatal_resistance = self.rc_min * (1.0 + 100.0 / max(1.0, solar * self.tau_cover))
            transpiration_w = (self.area_floor * self.lai * self.RHO_AIR * self.CP_AIR * (vpd / 0.066)) / (aerodynamic_resistance + stomatal_resistance)
            sensible = solar_net * 0.45 - conduction - sensible_vent - transpiration_w * 0.1
            w_inlet = 0.622 * (self.saturation_vapor_pressure(temp_inlet) * rh_inlet / 100.0) / 101.3
            w_internal = 0.622 * vapor_pressure / 101.3
            latent = transpiration_w / self.LAMBDA_LATENT - vent * self.RHO_AIR * (w_internal - w_inlet)
            return [sensible, latent]

        if SCIPY_AVAILABLE:
            solution = opt.root(residual, [temp_out + 4.0, min(95.0, rh_out + 12.0)], method="hybr")
            temp_in = float(solution.x[0])
            rh_in = float(np.clip(solution.x[1], 5.0, 100.0))
            solver_success = bool(solution.success)
        else:
            temp_in = temp_inlet + max(1.0, solar_net / max(self.area_floor, 1.0) / 125.0)
            rh_in = min(100.0, rh_inlet + 6.0)
            solver_success = False

        vent_final = self.ventilation_rate(temp_in, temp_out, wind)
        return {
            "internal_temperature_c": temp_in,
            "internal_relative_humidity_pct": rh_in,
            "ventilation_rate_m3_s": vent_final,
            "air_changes_per_hour": (vent_final * 3600.0) / max(self.volume, 1.0),
            "solver_success": solver_success,
        }


CROP_DATABASE: Dict[str, CropProfile] = {
    "Tomato - truss/cherry": CropProfile("Tomato", "fruiting vegetable", 34.0, 1.10, 8.0, 0.72, 0.75, "Inland low humidity", "High value, high cooling sensitivity"),
    "Cucumber - long": CropProfile("Cucumber", "fruiting vegetable", 46.0, 1.00, 5.8, 0.55, 0.58, "Inland central plains", "Fast cycles and strong ventilation demand"),
    "Sweet pepper": CropProfile("Sweet pepper", "fruiting vegetable", 24.0, 1.05, 12.0, 0.65, 0.82, "Low humidity inland", "Sensitive to heat stress and flower drop"),
    "Lettuce": CropProfile("Lettuce", "leafy green", 28.0, 0.78, 7.0, 0.38, 0.42, "Coastal controlled systems", "Short cycle, suitable for stacked production"),
    "Strawberry": CropProfile("Strawberry", "berry", 14.0, 0.90, 24.0, 0.78, 0.88, "Fully controlled/coastal", "High value but climate sensitive"),
    "Eggplant": CropProfile("Eggplant", "fruiting vegetable", 31.0, 0.98, 6.5, 0.58, 0.62, "Inland farms", "Robust crop with moderate value"),
    "Melon - netted": CropProfile("Melon", "vine crop", 18.0, 0.95, 9.5, 0.68, 0.68, "Inland large bays", "Needs space and careful humidity management"),
    "Basil and herbs": CropProfile("Basil and herbs", "herb", 18.5, 0.72, 18.0, 0.35, 0.45, "Controlled or semi-controlled", "High price, compact production"),
}


GREENHOUSE_TECHS: Dict[str, GreenhouseTech] = {
    "Low-tech shade net": GreenhouseTech(
        "Low-tech shade net",
        "low-tech",
        120,
        18,
        "passive",
        0.0,
        0.0,
        0.0,
        0.58,
        0.92,
        0.45,
        1.15,
        "Lowest capital cost, weak summer climate control; best only for tolerant crops and seasonal windows.",
    ),
    "Fan-pad evaporative": GreenhouseTech(
        "Fan-pad evaporative",
        "mid-tech",
        360,
        42,
        "evaporative",
        0.0,
        0.84,
        0.0,
        1.00,
        1.18,
        1.00,
        1.00,
        "Efficient inland where relative humidity is lower; high cooling water demand.",
    ),
    "Hybrid pad + chiller": GreenhouseTech(
        "Hybrid pad + chiller",
        "high-tech",
        1050,
        95,
        "hybrid",
        2.7,
        0.78,
        0.0,
        1.18,
        0.68,
        1.55,
        0.90,
        "Balanced system that shifts from evaporative cooling to mechanical cooling in humid periods.",
    ),
    "Mechanical chiller": GreenhouseTech(
        "Mechanical chiller",
        "high-tech",
        880,
        82,
        "mechanical",
        3.2,
        0.0,
        0.0,
        1.10,
        0.18,
        1.85,
        0.86,
        "Minimizes cooling water use but raises electrical demand and grid dependency.",
    ),
    "Fully controlled + LED": GreenhouseTech(
        "Fully controlled + LED",
        "CEA",
        1800,
        155,
        "mechanical",
        3.6,
        0.0,
        32.0,
        1.36,
        0.12,
        2.20,
        0.72,
        "Highest control, highest capital intensity; suited to premium crops and compact production.",
    ),
}


LAND_USE_COLORS = {
    "Agricultural": "#2f855a",
    "Open desert/rangeland": "#d6a84f",
    "Unclassified open land": "#c2a95f",
    "Residential": "#d43d3d",
    "Industrial": "#6b7280",
    "Protected area": "#7c3aed",
    "Flood-prone depression": "#f97316",
    "Urban expansion buffer": "#e11d48",
    "Water/offshore": "#2563eb",
}


def synthetic_lines(name: str) -> gpd.GeoDataFrame:
    if name == "power":
        geometries = [
            LineString([(51.03, 24.88), (51.14, 25.05), (51.28, 25.29), (51.47, 25.43)]),
            LineString([(50.93, 25.36), (51.17, 25.31), (51.42, 25.28)]),
            LineString([(51.12, 25.67), (51.25, 25.45), (51.35, 25.31)]),
            LineString([(51.00, 24.74), (51.22, 24.93), (51.55, 25.02)]),
        ]
    else:
        geometries = [
            LineString([(51.18, 24.72), (51.25, 25.05), (51.32, 25.35), (51.45, 25.71), (51.51, 26.02)]),
            LineString([(50.88, 25.22), (51.08, 25.25), (51.27, 25.31), (51.53, 25.37)]),
            LineString([(50.92, 25.10), (51.12, 25.05), (51.35, 25.04)]),
            LineString([(50.88, 25.60), (51.05, 25.54), (51.24, 25.45)]),
        ]
    return gpd.GeoDataFrame(
        {"name": [f"synthetic_{name}_{i + 1}" for i in range(len(geometries))]},
        geometry=geometries,
        crs=QATAR_CRS,
    )


def synthetic_landuse() -> gpd.GeoDataFrame:
    records = [
        ("Al Shahaniya agriculture", "Agricultural", True, Polygon([(50.88, 25.12), (51.20, 25.10), (51.22, 25.38), (50.90, 25.39)])),
        ("Rawdat Rashed agriculture", "Agricultural", True, Polygon([(50.93, 24.85), (51.22, 24.82), (51.24, 25.05), (50.96, 25.08)])),
        ("Umm Salal agriculture", "Agricultural", True, Polygon([(51.24, 25.38), (51.48, 25.37), (51.46, 25.61), (51.22, 25.60)])),
        ("Northern open land", "Open desert/rangeland", True, Polygon([(50.92, 25.62), (51.22, 25.58), (51.30, 25.92), (51.05, 26.05), (50.86, 25.86)])),
        ("Southern open land", "Open desert/rangeland", True, Polygon([(50.84, 24.66), (51.10, 24.62), (51.18, 24.82), (50.92, 24.95)])),
        ("Doha residential/urban", "Residential", False, Point(51.53, 25.29).buffer(0.105)),
        ("Al Wakrah residential", "Residential", False, Point(51.60, 25.17).buffer(0.060)),
        ("Al Khor residential", "Residential", False, Point(51.51, 25.68).buffer(0.060)),
        ("Umm Salal residential", "Residential", False, Point(51.40, 25.42).buffer(0.045)),
        ("Mesaieed industrial", "Industrial", False, Point(51.55, 24.99).buffer(0.070)),
        ("Dukhan industrial", "Industrial", False, Point(50.79, 25.42).buffer(0.060)),
        ("Al Reem protected area", "Protected area", False, Point(50.88, 25.67).buffer(0.115)),
        ("Rawda flood-prone depression", "Flood-prone depression", False, Point(51.10, 25.15).buffer(0.075)),
        ("Doha urban expansion buffer", "Urban expansion buffer", False, Point(51.42, 25.30).buffer(0.085)),
    ]
    return gpd.GeoDataFrame(
        {
            "name": [record[0] for record in records],
            "landuse": [record[1] for record in records],
            "greenhouse_ok": [record[2] for record in records],
        },
        geometry=[record[3] for record in records],
        crs=QATAR_CRS,
    )


def synthetic_groundwater_wells() -> gpd.GeoDataFrame:
    records = [
        ("GW-ALSHAHANIYA-01", "Synthetic hydrogeological well proxy", "Central Qatar / transition aquifer", Point(51.08, 25.33)),
        ("GW-RAWDAT-01", "Synthetic hydrogeological well proxy", "Central Qatar / transition aquifer", Point(51.08, 24.98)),
        ("GW-UMMSALAL-01", "Synthetic hydrogeological well proxy", "Northern aquifer / farm belt", Point(51.34, 25.50)),
        ("GW-NORTH-01", "Synthetic hydrogeological well proxy", "Northern aquifer / farm belt", Point(51.15, 25.78)),
        ("GW-SOUTH-01", "Synthetic hydrogeological well proxy", "Southern / southwestern stressed aquifer", Point(50.98, 24.78)),
        ("GW-MESAIEED-01", "Synthetic hydrogeological well proxy", "Coastal saline / industrial fringe", Point(51.42, 24.90)),
    ]
    profiles = [GROUNDWATER_REGIONAL_PROFILES[record[2]] for record in records]
    return gpd.GeoDataFrame(
        {
            "name": [record[0] for record in records],
            "subtype": [record[1] for record in records],
            "aquifer_zone": [record[2] for record in records],
            "source": [profile.source_note for profile in profiles],
            "source_tier": [profile.source_tier for profile in profiles],
            "tds_mg_l": [profile.tds_mg_l for profile in profiles],
            "water_level_m_bgl": [profile.water_level_m_bgl for profile in profiles],
            "well_yield_m3_day": [profile.well_yield_m3_day for profile in profiles],
            "permitted_abstraction_m3_year": [profile.permitted_abstraction_m3_year for profile in profiles],
            "transmissivity_m2_day": [profile.transmissivity_m2_day for profile in profiles],
            "storativity": [profile.storativity for profile in profiles],
            "aquifer_stress_ratio": [profile.aquifer_stress_ratio for profile in profiles],
            "groundwater_confidence": [profile.confidence for profile in profiles],
        },
        geometry=[record[3] for record in records],
        crs=QATAR_CRS,
    )


def off_land_result() -> dict:
    return {
        "feasible": False,
        "landuse": "Water/offshore",
        "landuse_name": "Outside the Qatar land mask",
        "groundwater_distance_m": float("inf"),
        "groundwater_score": 0.0,
        "groundwater_source": "Not applicable",
        "crop": "Not applicable",
        "technology": "Not applicable",
        "temp_c": np.nan,
        "rh_pct": np.nan,
        "ghi_w_m2": np.nan,
        "et0_mm_day": np.nan,
        "yield_tons": 0.0,
        "irrigation_m3": 0.0,
        "cooling_water_m3": 0.0,
        "total_water_m3": 0.0,
        "total_energy_mwh": 0.0,
        "peak_cooling_kw": 0.0,
        "water_l_kg": np.nan,
        "energy_kwh_kg": np.nan,
        "internal_temperature_c": np.nan,
        "internal_relative_humidity_pct": np.nan,
        "ventilation_rate_m3_s": 0.0,
        "air_changes_per_hour": 0.0,
        "microclimate_solver": False,
        "capital_qar": 0.0,
        "opex_qar": 0.0,
        "revenue_qar": 0.0,
        "net_profit_qar": 0.0,
        "payback_years": float("inf"),
        "roi_percent": 0.0,
    }


@st.cache_data(show_spinner=False)
def load_vector_layer(filename: str, fallback: str) -> gpd.GeoDataFrame:
    path = DATA_DIR / filename
    if path.exists():
        gdf = gpd.read_file(path).to_crs(QATAR_CRS)
        if fallback == "landuse":
            if "landuse" not in gdf.columns:
                gdf["landuse"] = gdf.get("class", "Unknown")
            if "greenhouse_ok" not in gdf.columns:
                allowed = {"agricultural", "agriculture", "farm", "open desert", "rangeland", "bare land"}
                gdf["greenhouse_ok"] = gdf["landuse"].astype(str).str.lower().isin(allowed)
        return gdf
    if fallback == "power":
        return synthetic_lines("power")
    if fallback == "roads":
        return synthetic_lines("roads")
    return synthetic_landuse()


@st.cache_data(show_spinner=False)
def load_groundwater_wells() -> gpd.GeoDataFrame:
    local_path = DATA_DIR / "qatar_groundwater_wells.geojson"
    if local_path.exists():
        wells = gpd.read_file(local_path).to_crs(QATAR_CRS)
        wells["source"] = wells.get("source", "Local qatar_groundwater_wells.geojson")
        wells["subtype"] = wells.get("subtype", wells.get("SUBTYPEDESCRIPTION", "Groundwater well"))
        wells["name"] = wells.get("name", wells.get("ASSETID", wells.index.astype(str)))
        return wells

    if ENABLE_REMOTE_GIS:
        try:
            wells = gpd.read_file(CGIS_GROUNDWATER_WELLS_URL).to_crs(QATAR_CRS)
            if not wells.empty:
                wells["name"] = wells.get("ASSETID", wells["OBJECTID"].astype(str) if "OBJECTID" in wells.columns else wells.index.astype(str))
                wells["subtype"] = wells.get("SUBTYPEDESCRIPTION", "HydrogeologicalStation")
                wells["source"] = "CGIS Qatar Vector/Water WATER.Facility HydrogeologicalStation"
                return wells
        except Exception:
            pass

    fallback = synthetic_groundwater_wells()
    fallback["source"] = fallback["source"] + "; fast local proxy. Replace with official qatar_groundwater_wells.geojson"
    return fallback


def allowed_landuse(landuse: str) -> bool:
    return landuse in {"Agricultural", "Open desert/rangeland", "Unclassified open land"}


def split_landuse(landuse: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    allowed = landuse[landuse["greenhouse_ok"].astype(bool)].copy()
    excluded = landuse[~landuse["greenhouse_ok"].astype(bool)].copy()
    return allowed, excluded


def point_distance_m(point: Point, layer: gpd.GeoDataFrame) -> float:
    if layer.empty:
        return float("inf")
    point_gdf = gpd.GeoDataFrame(geometry=[point], crs=QATAR_CRS).to_crs(QATAR_METRIC_CRS)
    projected = layer.to_crs(QATAR_METRIC_CRS)
    return float(point_gdf.geometry.iloc[0].distance(projected.geometry).min())


def nearest_feature_info(point: Point, layer: gpd.GeoDataFrame, name_field: str = "name") -> dict:
    if layer.empty:
        return {"distance_m": float("inf"), "name": "Unavailable", "source": "No groundwater layer"}
    point_gdf = gpd.GeoDataFrame(geometry=[point], crs=QATAR_CRS).to_crs(QATAR_METRIC_CRS)
    projected = layer.to_crs(QATAR_METRIC_CRS)
    distances = projected.geometry.distance(point_gdf.geometry.iloc[0])
    idx = distances.idxmin()
    feature = layer.loc[idx]
    return {
        "distance_m": float(distances.loc[idx]),
        "name": str(feature.get(name_field, feature.get("ASSETID", "Nearest groundwater feature"))),
        "source": str(feature.get("source", "Groundwater layer")),
        "attributes": {str(col): feature.get(col) for col in layer.columns if col != "geometry"},
    }


def polygon_distance_m(poly: Polygon, layer: gpd.GeoDataFrame) -> float:
    if layer.empty:
        return float("inf")
    poly_gdf = gpd.GeoDataFrame(geometry=[poly], crs=QATAR_CRS).to_crs(QATAR_METRIC_CRS)
    projected = layer.to_crs(QATAR_METRIC_CRS)
    return float(poly_gdf.geometry.iloc[0].distance(projected.geometry).min())


def point_landuse(point: Point, landuse: gpd.GeoDataFrame) -> dict:
    hits = landuse[landuse.intersects(point)]
    if hits.empty:
        return {"landuse": "Unclassified open land", "greenhouse_ok": True, "name": "Requires official land-use verification"}
    disallowed = hits[~hits["greenhouse_ok"].astype(bool)]
    selected = disallowed.iloc[0] if not disallowed.empty else hits.iloc[0]
    return {
        "landuse": str(selected.get("landuse", "Unknown")),
        "greenhouse_ok": bool(selected.get("greenhouse_ok", False)),
        "name": str(selected.get("name", "Land-use polygon")),
    }


def normalize_distance(distance_m: float, ideal_m: float, max_m: float) -> float:
    if distance_m <= ideal_m:
        return 1.0
    if distance_m >= max_m:
        return 0.0
    return 1.0 - ((distance_m - ideal_m) / (max_m - ideal_m))


def normalize_high(value: float, poor: float, good: float) -> float:
    if not np.isfinite(value):
        return 0.0
    if value <= poor:
        return 0.0
    if value >= good:
        return 1.0
    return (value - poor) / (good - poor)


def clamp01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def numeric_attr(attributes: dict, aliases: Iterable[str], default: float = np.nan) -> float:
    for alias in aliases:
        if alias in attributes:
            try:
                value = float(attributes[alias])
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                return value
    return default


def text_attr(attributes: dict, aliases: Iterable[str], default: str = "") -> str:
    for alias in aliases:
        value = attributes.get(alias)
        if value is not None and str(value).strip():
            return str(value)
    return default


def regional_groundwater_profile(lat: float, lon: float) -> GroundwaterRegionalProfile:
    if lon > 51.44 and lat < 25.34:
        return GROUNDWATER_REGIONAL_PROFILES["Coastal saline / industrial fringe"]
    if lat >= 25.40 and lon <= 51.48:
        return GROUNDWATER_REGIONAL_PROFILES["Northern aquifer / farm belt"]
    if lat < 25.05 or lon < 50.92:
        return GROUNDWATER_REGIONAL_PROFILES["Southern / southwestern stressed aquifer"]
    return GROUNDWATER_REGIONAL_PROFILES["Central Qatar / transition aquifer"]


def salinity_score(tds_mg_l: float) -> float:
    # Classes follow Qatar MAR/RWH studies using Ministry of Environment / Schlumberger 2009 salinity ranges.
    if not np.isfinite(tds_mg_l):
        return 0.35
    if tds_mg_l <= 1_000.0:
        return 1.0
    if tds_mg_l <= 2_500.0:
        return 0.82
    if tds_mg_l <= 4_000.0:
        return 0.62
    if tds_mg_l <= 5_000.0:
        return 0.42
    return max(0.08, 0.42 - (tds_mg_l - 5_000.0) / 10_000.0)


def water_level_score(depth_m_bgl: float) -> float:
    if not np.isfinite(depth_m_bgl):
        return 0.45
    return 0.20 + 0.80 * normalize_distance(depth_m_bgl, 18.0, 65.0)


def yield_score(well_yield_m3_day: float) -> float:
    if not np.isfinite(well_yield_m3_day):
        return 0.40
    return normalize_high(well_yield_m3_day, 60.0, 420.0)


def permit_score(permitted_abstraction_m3_year: float, source_confidence: float) -> float:
    if not np.isfinite(permitted_abstraction_m3_year):
        return 0.42 + 0.20 * source_confidence
    return normalize_high(permitted_abstraction_m3_year, 18_000.0, 90_000.0)


def transmissivity_score(transmissivity_m2_day: float) -> float:
    if not np.isfinite(transmissivity_m2_day):
        return 0.42
    return normalize_high(transmissivity_m2_day, 25.0, 500.0)


def storativity_score(storativity: float) -> float:
    if not np.isfinite(storativity) or storativity <= 0.0:
        return 0.40
    log_value = math.log10(storativity)
    return clamp01((log_value - math.log10(1e-5)) / (math.log10(1e-1) - math.log10(1e-5)))


def aquifer_stress_score(stress_ratio: float) -> float:
    if not np.isfinite(stress_ratio):
        stress_ratio = NATIONAL_GW_STRESS_RATIO
    if stress_ratio <= 1.0:
        return 1.0
    if stress_ratio >= 5.0:
        return 0.12
    return 1.0 - ((stress_ratio - 1.0) / 4.0) * 0.88


def groundwater_quality_model(lat: float, lon: float, nearest_info: dict) -> dict:
    attributes = nearest_info.get("attributes", {})
    profile = regional_groundwater_profile(lat, lon)
    confidence = numeric_attr(attributes, ["groundwater_confidence", "confidence"], profile.confidence)

    tds = numeric_attr(attributes, ["tds_mg_l", "TDS_MG_L", "TDS", "tds", "salinity_mg_l"], profile.tds_mg_l)
    water_level = numeric_attr(attributes, ["water_level_m_bgl", "depth_to_water_m", "gw_level_m_bgl", "waterlevel_m"], profile.water_level_m_bgl)
    well_yield = numeric_attr(attributes, ["well_yield_m3_day", "yield_m3_day", "safe_yield_m3_day", "pump_test_m3_day"], profile.well_yield_m3_day)
    permit = numeric_attr(attributes, ["permitted_abstraction_m3_year", "permit_m3_year", "allocation_m3_year"], profile.permitted_abstraction_m3_year)
    transmissivity = numeric_attr(attributes, ["transmissivity_m2_day", "transmissivity", "tm_m2_day"], profile.transmissivity_m2_day)
    storativity = numeric_attr(attributes, ["storativity", "storage_coefficient"], profile.storativity)
    stress_ratio = numeric_attr(attributes, ["aquifer_stress_ratio", "stress_ratio", "abstraction_safe_yield_ratio"], profile.aquifer_stress_ratio)

    exact_fields = sum(
        np.isfinite(numeric_attr(attributes, [field], np.nan))
        for field in GW_REQUIRED_FIELDS
    )
    source_tier = text_attr(attributes, ["source_tier", "tier"], profile.source_tier)
    source_note = text_attr(attributes, ["source_note", "source"], profile.source_note)

    distance_score = normalize_distance(float(nearest_info["distance_m"]), 2_000.0, 18_000.0)
    quality = salinity_score(tds)
    depth = water_level_score(water_level)
    quantity = yield_score(well_yield)
    permission = permit_score(permit, confidence)
    capacity = 0.62 * transmissivity_score(transmissivity) + 0.38 * storativity_score(storativity)
    stress = aquifer_stress_score(stress_ratio)
    raw_score = (
        distance_score * 0.18
        + quality * 0.20
        + depth * 0.10
        + quantity * 0.14
        + permission * 0.10
        + capacity * 0.13
        + stress * 0.15
    )
    data_completeness = exact_fields / len(GW_REQUIRED_FIELDS)
    uncertainty_penalty = 0.86 + 0.14 * max(data_completeness, confidence)
    composite = clamp01(raw_score * uncertainty_penalty)

    return {
        "score": composite,
        "distance_score": distance_score,
        "salinity_score": quality,
        "water_level_score": depth,
        "well_yield_score": quantity,
        "permit_score": permission,
        "capacity_score": capacity,
        "stress_score": stress,
        "tds_mg_l": tds,
        "water_level_m_bgl": water_level,
        "well_yield_m3_day": well_yield,
        "permitted_abstraction_m3_year": permit,
        "transmissivity_m2_day": transmissivity,
        "storativity": storativity,
        "aquifer_stress_ratio": stress_ratio,
        "aquifer_zone": text_attr(attributes, ["aquifer_zone"], profile.zone),
        "source_tier": source_tier,
        "source_note": source_note,
        "confidence": confidence,
        "data_completeness": data_completeness,
    }


def interpolate_climate(lat: float, lon: float) -> dict:
    point = Point(lon, lat)
    if not QATAR_POLYGON.contains(point):
        return {"temp_c": np.nan, "rh_pct": np.nan, "ghi_w_m2": np.nan, "station_note": "water/offshore"}

    weighted = []
    for name, (station_lon, station_lat, temp_c, rh_pct, ghi_w_m2) in REFERENCE_STATIONS.items():
        distance = max(math.hypot(lon - station_lon, lat - station_lat), 0.0001)
        weight = 1.0 / (distance**2)
        weighted.append((weight, name, temp_c, rh_pct, ghi_w_m2))

    weight_sum = sum(row[0] for row in weighted)
    temp = sum(weight * temp for weight, _, temp, _, _ in weighted) / weight_sum
    rh = sum(weight * rh for weight, _, _, rh, _ in weighted) / weight_sum
    ghi = sum(weight * ghi for weight, _, _, _, ghi in weighted) / weight_sum
    nearest = min(weighted, key=lambda row: 1.0 / math.sqrt(row[0]))[1]
    return {"temp_c": round(temp, 1), "rh_pct": round(rh, 1), "ghi_w_m2": round(ghi, 0), "station_note": f"IDW; nearest {nearest}"}


def et0_hargreaves_mm_day(temp_c: float, ghi_w_m2: float) -> float:
    if not np.isfinite(temp_c) or not np.isfinite(ghi_w_m2):
        return 0.0
    solar_mj_m2_day = ghi_w_m2 * 12.0 * 3600.0 / 1_000_000.0
    et0 = 0.0023 * solar_mj_m2_day * math.sqrt(12.0) * (temp_c + 17.8)
    return round(max(1.5, et0), 2)


def cooling_load_kw(area_m2: float, transmissivity: float, climate: dict, tech: GreenhouseTech) -> float:
    if not np.isfinite(climate["temp_c"]) or not np.isfinite(climate["ghi_w_m2"]):
        return 0.0
    solar_gain_w = area_m2 * climate["ghi_w_m2"] * transmissivity * 0.88
    delta_t = max(0.0, climate["temp_c"] - T_SET_C)
    sensible_w = 1.4 * (area_m2 * 0.55) * delta_t
    humidity_penalty = max(0.0, climate["rh_pct"] - 55.0) / 100.0
    latent_w = (solar_gain_w + sensible_w) * humidity_penalty * (0.55 if tech.cooling_mode in {"evaporative", "hybrid"} else 0.25)
    return (solar_gain_w + sensible_w + latent_w) / 1000.0


def greenhouse_geometry(area_m2: int, transmissivity: float, crop: CropProfile) -> dict:
    bay_width = 9.6
    length = max(24.0, area_m2 / bay_width)
    cover_factor = 1.78
    return {
        "length": length,
        "width": bay_width,
        "height_eaves": 4.0,
        "height_roof": 5.6,
        "area_cover": area_m2 * cover_factor,
        "area_vent_roof": area_m2 * 0.065,
        "area_vent_side": area_m2 * 0.095,
        "tau_cover": transmissivity,
        "net_porosity": 0.52,
        "lai": 2.6 if crop.crop_family == "fruiting vegetable" else 1.8,
        "rc_min": 120.0 if crop.crop_family == "fruiting vegetable" else 95.0,
    }


def control_aware_microclimate_state(
    area_m2: int,
    transmissivity: float,
    crop: CropProfile,
    tech: GreenhouseTech,
    climate: dict,
    detailed: bool,
) -> dict:
    """Stable planning-grade indoor climate estimate with explicit cooling controls.

    The original nonlinear ventilation balance is useful for naturally ventilated
    envelopes, but it does not include an active chiller control term. This wrapper
    keeps the dashboard monotonic: higher transmissivity increases heat load, larger
    houses alter both temperature and ACH, and mechanical systems pull the result
    toward the crop setpoint.
    """
    if not np.isfinite(climate["temp_c"]):
        return {
            "internal_temperature_c": np.nan,
            "internal_relative_humidity_pct": np.nan,
            "ventilation_rate_m3_s": 0.0,
            "air_changes_per_hour": 0.0,
            "solver_success": False,
        }

    temp_out = float(climate["temp_c"])
    rh_out = float(climate["rh_pct"])
    ghi = float(climate["ghi_w_m2"])
    area = max(float(area_m2), 500.0)
    engine = AdvancedGreenhouseEngine(greenhouse_geometry(int(area), transmissivity, crop))

    solar_pressure = np.clip((ghi * transmissivity) / (890.0 * 0.65), 0.55, 1.55)
    area_pressure = np.clip(math.log2(area / 5000.0), -1.6, 2.2)
    heat_load = (solar_pressure - 1.0) + 0.22 * area_pressure
    humidity_penalty = max(0.0, rh_out - 55.0)
    detail_offset = 0.0 if detailed else 0.35

    pad_temp, pad_rh = engine.evaporative_pad_outlet(temp_out, rh_out, max(tech.pad_efficiency, 0.84))

    if tech.cooling_mode == "mechanical":
        cop_bonus = max(0.0, tech.cop - 3.0) * 0.35
        led_bonus = 0.45 if tech.name == "Fully controlled + LED" else 0.0
        temp_in = T_SET_C + 1.05 + heat_load * 1.05 + humidity_penalty * 0.014 - cop_bonus - led_bonus + detail_offset
        rh_in = np.clip(rh_out + 2.5 - led_bonus * 6.0, 45.0, 72.0)
        vent_coeff = 0.00185 if tech.name == "Fully controlled + LED" else 0.00220
    elif tech.cooling_mode == "hybrid":
        evap_temp = pad_temp + 2.0 + heat_load * 2.25 + humidity_penalty * 0.040
        chiller_temp = T_SET_C + 1.45 + heat_load * 0.95 + humidity_penalty * 0.018
        temp_in = min(evap_temp, chiller_temp) + detail_offset
        rh_in = np.clip((pad_rh + rh_out) / 2.0 + 5.0, 50.0, 86.0)
        vent_coeff = 0.00310
    elif tech.cooling_mode == "evaporative":
        temp_in = pad_temp + 2.6 + heat_load * 2.75 + humidity_penalty * 0.060 + crop.climate_sensitivity * 0.55 + detail_offset
        rh_in = np.clip(pad_rh + 4.0, max(rh_out, 35.0), 96.0)
        vent_coeff = 0.00380
    else:
        temp_in = temp_out + 3.8 + heat_load * 4.70 + crop.climate_sensitivity * 1.05 + detail_offset
        rh_in = np.clip(rh_out - 3.0, 18.0, 86.0)
        vent_coeff = 0.00160

    # Sub-linear fan/vent scaling makes ACH respond to greenhouse size instead of
    # cancelling out exactly when area and volume grow together.
    fan_response = np.clip(1.0 + heat_load * (0.12 if tech.cooling_mode == "passive" else 0.18), 0.82, 1.28)
    vent_rate = max(0.6, vent_coeff * (area**0.88) * (5000.0**0.12) * fan_response)
    volume = max(area * 4.8, 1.0)
    return {
        "internal_temperature_c": float(np.clip(temp_in, 18.0, temp_out + 9.0)),
        "internal_relative_humidity_pct": float(rh_in),
        "ventilation_rate_m3_s": float(vent_rate),
        "air_changes_per_hour": float((vent_rate * 3600.0) / volume),
        "solver_success": bool(detailed),
    }


def microclimate_design_state(area_m2: int, transmissivity: float, crop: CropProfile, tech: GreenhouseTech, climate: dict) -> dict:
    return control_aware_microclimate_state(area_m2, transmissivity, crop, tech, climate, detailed=True)


def approximate_microclimate_state(area_m2: int, transmissivity: float, crop: CropProfile, tech: GreenhouseTech, climate: dict) -> dict:
    return control_aware_microclimate_state(area_m2, transmissivity, crop, tech, climate, detailed=False)


def analyze_location(
    lat: float,
    lon: float,
    crop: CropProfile,
    tech: GreenhouseTech,
    area_m2: int,
    transmissivity: float,
    recycle_drainage: bool,
    landuse: gpd.GeoDataFrame,
    include_microclimate: bool = True,
) -> dict:
    point = Point(lon, lat)
    if not QATAR_POLYGON.contains(point):
        result = off_land_result()
        result["crop"] = crop.name
        result["technology"] = tech.name
        return result
    lu = point_landuse(point, landuse)
    climate = interpolate_climate(lat, lon)
    et0 = et0_hargreaves_mm_day(climate["temp_c"], climate["ghi_w_m2"])
    irrigation_m3 = (et0 * crop.kc / 1000.0) * area_m2 * 365.0
    if recycle_drainage:
        irrigation_m3 *= 0.70

    peak_kw = cooling_load_kw(area_m2, transmissivity, climate, tech)
    rh_penalty = 1.0 + max(0.0, (climate["rh_pct"] - 55.0) / 55.0)

    if tech.cooling_mode == "passive":
        cooling_water_m3 = 0.04 * peak_kw * COOLING_HOURS_YEAR / 1000.0
        cooling_energy_mwh = area_m2 * 12.0 / 1000.0
    elif tech.cooling_mode == "evaporative":
        cooling_water_m3 = peak_kw * COOLING_HOURS_YEAR * 0.00115 * rh_penalty / max(tech.pad_efficiency, 0.1)
        fan_kw = area_m2 * 0.0042
        pump_kw = area_m2 * 0.0010
        cooling_energy_mwh = (fan_kw + pump_kw) * COOLING_HOURS_YEAR / 1000.0
    elif tech.cooling_mode == "hybrid":
        evap_fraction = max(0.20, min(0.70, 1.0 - (climate["rh_pct"] - 45.0) / 45.0))
        evap_water = peak_kw * evap_fraction * COOLING_HOURS_YEAR * 0.00095 * rh_penalty / max(tech.pad_efficiency, 0.1)
        chiller_energy = peak_kw * (1.0 - evap_fraction) * COOLING_HOURS_YEAR / max(tech.cop, 0.1) / 1000.0
        fan_energy = area_m2 * 0.0035 * COOLING_HOURS_YEAR / 1000.0
        cooling_water_m3 = evap_water
        cooling_energy_mwh = chiller_energy + fan_energy
    else:
        cooling_water_m3 = peak_kw * COOLING_HOURS_YEAR * 0.00008
        cooling_energy_mwh = peak_kw * COOLING_HOURS_YEAR / max(tech.cop, 0.1) / 1000.0

    base_energy_mwh = area_m2 * 34.0 / 1000.0
    lighting_mwh = tech.lighting_w_m2 * area_m2 * 2000.0 / 1_000_000.0
    total_energy_mwh = (cooling_energy_mwh + base_energy_mwh + lighting_mwh) * tech.energy_multiplier
    total_water_m3 = (irrigation_m3 + cooling_water_m3) * tech.water_multiplier
    if include_microclimate:
        micro_state = microclimate_design_state(area_m2, transmissivity, crop, tech, climate)
    else:
        micro_state = approximate_microclimate_state(area_m2, transmissivity, crop, tech, climate)

    stress = 1.0
    if climate["temp_c"] > 40.0 and tech.cooling_mode in {"passive", "evaporative"}:
        stress -= crop.climate_sensitivity * 0.16
    if climate["rh_pct"] > 60.0 and tech.cooling_mode == "evaporative":
        stress -= crop.climate_sensitivity * 0.12
    if not lu["greenhouse_ok"]:
        stress = 0.0
    stress = max(0.0, stress)

    yield_kg = crop.yield_kg_m2_year * area_m2 * tech.yield_multiplier * stress
    revenue_qar = yield_kg * crop.price_qar_kg
    capital_qar = area_m2 * tech.capital_qar_m2
    fixed_opex_qar = area_m2 * tech.fixed_opex_qar_m2_year
    electricity_qar = total_energy_mwh * 1000.0 * ELECTRICITY_QAR_KWH
    water_qar = total_water_m3 * WATER_QAR_M3
    labour_qar = (area_m2 / 1000.0) * 0.12 * 5000.0 * 12.0 * tech.labour_factor
    maintenance_qar = capital_qar * 0.035
    total_opex_qar = fixed_opex_qar + electricity_qar + water_qar + labour_qar + maintenance_qar
    net_profit_qar = revenue_qar - total_opex_qar
    payback_years = capital_qar / net_profit_qar if net_profit_qar > 0 else float("inf")
    roi_percent = net_profit_qar / capital_qar * 100.0 if capital_qar > 0 else 0.0

    return {
        "feasible": bool(lu["greenhouse_ok"]),
        "landuse": lu["landuse"],
        "landuse_name": lu["name"],
        "crop": crop.name,
        "technology": tech.name,
        "temp_c": climate["temp_c"],
        "rh_pct": climate["rh_pct"],
        "ghi_w_m2": climate["ghi_w_m2"],
        "et0_mm_day": et0,
        "yield_tons": yield_kg / 1000.0,
        "irrigation_m3": irrigation_m3,
        "cooling_water_m3": cooling_water_m3,
        "total_water_m3": total_water_m3,
        "total_energy_mwh": total_energy_mwh,
        "peak_cooling_kw": peak_kw,
        "water_l_kg": (total_water_m3 * 1000.0 / yield_kg) if yield_kg > 0 else float("inf"),
        "energy_kwh_kg": (total_energy_mwh * 1000.0 / yield_kg) if yield_kg > 0 else float("inf"),
        "internal_temperature_c": micro_state["internal_temperature_c"],
        "internal_relative_humidity_pct": micro_state["internal_relative_humidity_pct"],
        "ventilation_rate_m3_s": micro_state["ventilation_rate_m3_s"],
        "air_changes_per_hour": micro_state["air_changes_per_hour"],
        "microclimate_solver": micro_state["solver_success"],
        "capital_qar": capital_qar,
        "opex_qar": total_opex_qar,
        "revenue_qar": revenue_qar,
        "net_profit_qar": net_profit_qar,
        "payback_years": payback_years,
        "roi_percent": roi_percent,
    }


def calculate_suitability(lat: float, lon: float, weights: dict, layers: dict) -> dict:
    point = Point(lon, lat)
    if not QATAR_POLYGON.contains(point):
        return {
            "score": 0.0,
            "status": "Water/offshore: outside the Qatar land mask",
            "is_excluded": True,
            "landuse": "Water/offshore",
            "landuse_name": "No greenhouse analysis is calculated for water",
            "landuse_score": 0.0,
            "grid_distance_m": float("inf"),
            "road_distance_m": float("inf"),
            "groundwater_distance_m": float("inf"),
            "groundwater_source": "Not applicable",
            "groundwater_source_tier": "Not applicable",
            "groundwater_aquifer_zone": "Not applicable",
            "groundwater_tds_mg_l": np.nan,
            "groundwater_level_m_bgl": np.nan,
            "groundwater_well_yield_m3_day": np.nan,
            "groundwater_permitted_abstraction_m3_year": np.nan,
            "groundwater_transmissivity_m2_day": np.nan,
            "groundwater_storativity": np.nan,
            "groundwater_aquifer_stress_ratio": np.nan,
            "groundwater_confidence": 0.0,
            "groundwater_data_completeness": 0.0,
            "groundwater_distance_score": 0.0,
            "groundwater_salinity_score": 0.0,
            "groundwater_level_score": 0.0,
            "groundwater_yield_score": 0.0,
            "groundwater_permit_score": 0.0,
            "groundwater_capacity_score": 0.0,
            "groundwater_stress_score": 0.0,
            "excluded_distance_m": float("inf"),
            "climate_score": 0.0,
            "grid_score": 0.0,
            "road_score": 0.0,
            "groundwater_score": 0.0,
            "constraint_score": 0.0,
        }

    climate = interpolate_climate(lat, lon)
    lu = point_landuse(point, layers["landuse"])
    grid_distance = point_distance_m(point, layers["power"])
    road_distance = point_distance_m(point, layers["roads"])
    groundwater_info = nearest_feature_info(point, layers["groundwater_wells"])
    groundwater_distance = groundwater_info["distance_m"]
    groundwater_model = groundwater_quality_model(lat, lon, groundwater_info)
    excluded_distance = point_distance_m(point, layers["excluded_landuse"])

    climate_score = max(0.0, min(1.0, 1.0 - (climate["rh_pct"] - 38.0) / 35.0))
    grid_score = normalize_distance(grid_distance, 800.0, 28_000.0)
    road_score = normalize_distance(road_distance, 1_200.0, 24_000.0)
    groundwater_score = groundwater_model["score"]
    constraint_score = min(1.0, excluded_distance / 1500.0)
    if lu["greenhouse_ok"]:
        if lu["landuse"] == "Agricultural":
            landuse_score = 1.0
        elif lu["landuse"] == "Unclassified open land":
            landuse_score = 0.58
        else:
            landuse_score = 0.72
    else:
        landuse_score = 0.0

    raw_score = (
        climate_score * weights["climate"]
        + grid_score * weights["grid"]
        + road_score * weights["logistics"]
        + groundwater_score * weights["groundwater"]
        + landuse_score * weights["landuse"]
        + constraint_score * weights["constraints"]
    )
    score = round(max(0.0, min(raw_score * 100.0, 100.0)), 1)
    is_excluded = not lu["greenhouse_ok"]
    if is_excluded:
        score = 0.0
        status = f"Excluded land use: {lu['landuse']}"
    elif score >= 78:
        status = "Excellent: realistic greenhouse candidate"
    elif score >= 58:
        status = "Good: viable after site verification"
    elif score >= 38:
        status = "Moderate: resource or infrastructure burden"
    else:
        status = "Low suitability"

    return {
        "score": score,
        "status": status,
        "is_excluded": is_excluded,
        "landuse": lu["landuse"],
        "landuse_name": lu["name"],
        "grid_distance_m": grid_distance,
        "road_distance_m": road_distance,
        "groundwater_distance_m": groundwater_distance,
        "groundwater_source": groundwater_info["source"],
        "nearest_groundwater": groundwater_info["name"],
        "groundwater_source_tier": groundwater_model["source_tier"],
        "groundwater_aquifer_zone": groundwater_model["aquifer_zone"],
        "groundwater_tds_mg_l": groundwater_model["tds_mg_l"],
        "groundwater_level_m_bgl": groundwater_model["water_level_m_bgl"],
        "groundwater_well_yield_m3_day": groundwater_model["well_yield_m3_day"],
        "groundwater_permitted_abstraction_m3_year": groundwater_model["permitted_abstraction_m3_year"],
        "groundwater_transmissivity_m2_day": groundwater_model["transmissivity_m2_day"],
        "groundwater_storativity": groundwater_model["storativity"],
        "groundwater_aquifer_stress_ratio": groundwater_model["aquifer_stress_ratio"],
        "groundwater_confidence": groundwater_model["confidence"],
        "groundwater_data_completeness": groundwater_model["data_completeness"],
        "groundwater_distance_score": round(groundwater_model["distance_score"] * 100.0, 1),
        "groundwater_salinity_score": round(groundwater_model["salinity_score"] * 100.0, 1),
        "groundwater_level_score": round(groundwater_model["water_level_score"] * 100.0, 1),
        "groundwater_yield_score": round(groundwater_model["well_yield_score"] * 100.0, 1),
        "groundwater_permit_score": round(groundwater_model["permit_score"] * 100.0, 1),
        "groundwater_capacity_score": round(groundwater_model["capacity_score"] * 100.0, 1),
        "groundwater_stress_score": round(groundwater_model["stress_score"] * 100.0, 1),
        "excluded_distance_m": excluded_distance,
        "climate_score": round(climate_score * 100.0, 1),
        "grid_score": round(grid_score * 100.0, 1),
        "road_score": round(road_score * 100.0, 1),
        "groundwater_score": round(groundwater_score * 100.0, 1),
        "landuse_score": round(landuse_score * 100.0, 1),
        "constraint_score": round(constraint_score * 100.0, 1),
    }


def score_color(score: float, excluded: bool = False) -> str:
    if excluded:
        return "#991b1b"
    if score >= 78:
        return "#16803c"
    if score >= 58:
        return "#73a827"
    if score >= 38:
        return "#d18c00"
    return "#b33430"


@st.cache_data(show_spinner=False, ttl=1800)
def build_heatmap_runtime(weight_key: tuple[float, ...], resolution: int, _layers: dict) -> gpd.GeoDataFrame:
    weights = {
        "climate": weight_key[0],
        "grid": weight_key[1],
        "logistics": weight_key[2],
        "groundwater": weight_key[3],
        "landuse": weight_key[4],
        "constraints": weight_key[5],
    }
    records = []
    cell_radius = 0.48 / max(resolution, 1)
    for lat in np.linspace(24.68, 26.02, resolution):
        for lon in np.linspace(50.84, 51.55, resolution):
            point = Point(lon, lat)
            if not QATAR_POLYGON.contains(point):
                continue
            result = calculate_suitability(lat, lon, weights, _layers)
            records.append(
                {
                    "score": result["score"],
                    "status": result["status"],
                    "landuse": result["landuse"],
                    "groundwater_km": round(result["groundwater_distance_m"] / 1000.0, 1) if np.isfinite(result["groundwater_distance_m"]) else None,
                    "is_excluded": result["is_excluded"],
                    "geometry": point.buffer(cell_radius),
                }
            )
    return gpd.GeoDataFrame(records, crs=QATAR_CRS)


def add_geojson(map_object: folium.Map, gdf: gpd.GeoDataFrame, name: str, color: str, fill_color: Optional[str] = None, weight: int = 2, fill_opacity: float = 0.16) -> None:
    if gdf.empty:
        return
    folium.GeoJson(
        gdf,
        name=name,
        tooltip=folium.GeoJsonTooltip(fields=[col for col in ["name", "landuse", "greenhouse_ok"] if col in gdf.columns]),
        style_function=lambda _feature: {
            "color": color,
            "weight": weight,
            "fillColor": fill_color or color,
            "fillOpacity": fill_opacity,
        },
    ).add_to(map_object)


def add_groundwater_wells(map_object: folium.Map, wells: gpd.GeoDataFrame) -> None:
    if wells.empty:
        return
    group = folium.FeatureGroup(name="Groundwater / hydrogeological stations", show=True)
    for _, row in wells.iterrows():
        geometry = row.geometry
        if geometry is None or geometry.is_empty:
            continue
        folium.CircleMarker(
            location=[geometry.y, geometry.x],
            radius=5,
            color="#0f766e",
            fill=True,
            fill_color="#14b8a6",
            fill_opacity=0.85,
            tooltip=f"{row.get('name', 'Groundwater feature')} | {row.get('subtype', 'HydrogeologicalStation')}",
        ).add_to(group)
    group.add_to(map_object)


def popup_value(value: object, suffix: str = "", missing: str = "N/A", decimals: int = 1) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        text = str(value).strip() if value is not None else ""
        return html.escape(text) if text else missing
    if not np.isfinite(numeric):
        return missing
    if abs(numeric) >= 1000:
        rendered = f"{numeric:,.0f}"
    else:
        rendered = f"{numeric:,.{decimals}f}"
    return f"{rendered}{suffix}"


def selected_marker_tooltip(suitability: Optional[dict], climate: Optional[dict]) -> str:
    if not suitability:
        return "Selected greenhouse site"
    status = suitability.get("status", "Selected site")
    score = suitability.get("score", 0)
    landuse_value = suitability.get("landuse", "Unknown land use")
    rh = climate.get("rh_pct") if climate else np.nan
    temp = climate.get("temp_c") if climate else np.nan
    groundwater = suitability.get("groundwater_score", 0)
    status_color = "#166534" if not suitability.get("is_excluded") else "#991b1b"
    return f"""
    <div style="font-family:Arial,sans-serif;width:220px;white-space:normal;line-height:1.22;">
      <div style="font-size:12px;font-weight:700;color:{status_color};margin-bottom:3px;">Selected Site</div>
      <div style="display:flex;justify-content:space-between;gap:8px;font-size:11px;">
        <span style="color:#475569;">Suitability</span><b style="color:#0f172a;">{popup_value(score, '/100', decimals=0)}</b>
      </div>
      <div style="display:flex;justify-content:space-between;gap:8px;font-size:11px;">
        <span style="color:#475569;">Land use</span><b style="color:#0f172a;text-align:right;">{html.escape(str(landuse_value))}</b>
      </div>
      <div style="display:flex;justify-content:space-between;gap:8px;font-size:11px;">
        <span style="color:#475569;">Climate</span><b style="color:#0f172a;">{popup_value(temp, ' °C')} / {popup_value(rh, '% RH')}</b>
      </div>
      <div style="display:flex;justify-content:space-between;gap:8px;font-size:11px;">
        <span style="color:#475569;">Groundwater</span><b style="color:#0f172a;">{popup_value(groundwater, '/100', decimals=0)}</b>
      </div>
      <div style="font-size:10px;color:#64748b;margin-top:4px;border-top:1px solid #e2e8f0;padding-top:3px;">
        {html.escape(str(status))}
      </div>
    </div>
    """


def selected_marker_popup_html(suitability: Optional[dict], climate: Optional[dict], report: Optional[dict]) -> str:
    if not suitability:
        return "<b>Selected greenhouse site</b>"
    escaped_status = html.escape(str(suitability.get("status", "Selected site")))
    status_color = "#166534" if not suitability.get("is_excluded") else "#991b1b"
    landuse_value = html.escape(str(suitability.get("landuse", "Unknown")))
    landuse_name = html.escape(str(suitability.get("landuse_name", "")))
    aquifer = html.escape(str(suitability.get("groundwater_aquifer_zone", "Not available")))
    tier = html.escape(str(suitability.get("groundwater_source_tier", "Not available")))
    nearest_gw = html.escape(str(suitability.get("nearest_groundwater", "Not available")))

    rows = [
        ("Suitability", f"{popup_value(suitability.get('score'), '/100', decimals=0)}"),
        ("Status", escaped_status),
        ("Land use", landuse_value),
        ("Climate", f"{popup_value(climate.get('temp_c') if climate else np.nan, ' °C')} / {popup_value(climate.get('rh_pct') if climate else np.nan, '% RH')}"),
        ("Solar", popup_value(climate.get("ghi_w_m2") if climate else np.nan, " W/m²", decimals=0)),
        ("Grid", format_distance(float(suitability.get("grid_distance_m", float("inf"))))),
        ("Road", format_distance(float(suitability.get("road_distance_m", float("inf"))))),
        ("Groundwater", f"{popup_value(suitability.get('groundwater_score'), '/100', decimals=0)} | {format_distance(float(suitability.get('groundwater_distance_m', float('inf'))))}"),
        ("TDS / salinity", popup_value(suitability.get("groundwater_tds_mg_l"), " mg/L", decimals=0)),
        ("Aquifer stress", popup_value(suitability.get("groundwater_aquifer_stress_ratio"), "x safe yield", decimals=1)),
        ("Aquifer zone", aquifer),
        ("Evidence tier", tier),
    ]

    if report:
        rows.extend(
            [
                ("Tomato + fan-pad yield", popup_value(report.get("yield_tons"), " t/yr", decimals=1)),
                ("Water", popup_value(report.get("total_water_m3"), " m³/yr", decimals=0)),
                ("Energy", popup_value(report.get("total_energy_mwh"), " MWh/yr", decimals=0)),
                ("Net profit", f"QAR {popup_value(report.get('net_profit_qar'), decimals=0)}"),
            ]
        )

    row_html = "\n".join(
        f"<tr><td style='color:#475569;padding:3px 8px 3px 0;'>{html.escape(label)}</td>"
        f"<td style='font-weight:600;color:#0f172a;padding:3px 0;'>{value}</td></tr>"
        for label, value in rows
    )
    note = "No-build location: analysis is diagnostic only." if suitability.get("is_excluded") else "Planning-grade screening result."
    return f"""
    <div style="font-family:Arial,sans-serif;width:330px;line-height:1.25;">
      <div style="font-size:15px;font-weight:700;color:{status_color};margin-bottom:4px;">Selected Greenhouse Site</div>
      <div style="font-size:12px;color:#475569;margin-bottom:7px;">{landuse_name}</div>
      <table style="border-collapse:collapse;font-size:12px;width:100%;">{row_html}</table>
      <div style="font-size:11px;color:#64748b;margin-top:8px;border-top:1px solid #e2e8f0;padding-top:6px;">
        Nearest groundwater: {nearest_gw}. {html.escape(note)}
      </div>
    </div>
    """


def build_map(
    lat: float,
    lon: float,
    weights: dict,
    layers: dict,
    show_heatmap: bool,
    show_infra: bool,
    show_landuse: bool,
    show_groundwater: bool,
    heatmap_resolution: int,
    suitability: Optional[dict] = None,
    climate: Optional[dict] = None,
    selected_report: Optional[dict] = None,
) -> folium.Map:
    map_object = folium.Map(location=[25.3548, 51.1839], zoom_start=9, tiles="CartoDB positron", control_scale=True)
    folium.Rectangle(
        bounds=[[24.48, 50.66], [26.22, 51.72]],
        color="#60a5fa",
        weight=0,
        fill=True,
        fill_color="#dbeafe",
        fill_opacity=0.22,
        tooltip="Water/offshore areas are excluded from greenhouse analysis",
    ).add_to(map_object)
    boundary = gpd.GeoDataFrame({"name": ["Qatar analysis boundary"]}, geometry=[QATAR_POLYGON], crs=QATAR_CRS)
    add_geojson(map_object, boundary, "Qatar land mask", "#374151", fill_color="#f9fafb", fill_opacity=0.42)

    if show_heatmap:
        weight_key = tuple(round(weights[key], 4) for key in ["climate", "grid", "logistics", "groundwater", "landuse", "constraints"])
        heatmap = build_heatmap_runtime(weight_key, heatmap_resolution, layers)
        folium.GeoJson(
            heatmap,
            name="Feasible suitability heatmap",
            tooltip=folium.GeoJsonTooltip(fields=["score", "status", "landuse", "groundwater_km"]),
            style_function=lambda feature: {
                "fillColor": score_color(float(feature["properties"]["score"]), bool(feature["properties"]["is_excluded"])),
                "color": score_color(float(feature["properties"]["score"]), bool(feature["properties"]["is_excluded"])),
                "weight": 0.35,
                "fillOpacity": 0.46 if not feature["properties"]["is_excluded"] else 0.62,
            },
        ).add_to(map_object)

    if show_landuse:
        for landuse_name, color in LAND_USE_COLORS.items():
            subset = layers["landuse"][layers["landuse"]["landuse"] == landuse_name]
            opacity = 0.18 if allowed_landuse(landuse_name) else 0.34
            add_geojson(map_object, subset, f"Land use: {landuse_name}", color, fill_color=color, weight=2, fill_opacity=opacity)

    if show_infra:
        add_geojson(map_object, layers["power"], "Power grid proxy", "#c0262d", weight=4)
        add_geojson(map_object, layers["roads"], "Primary highways", "#2563eb", weight=4)

    if show_groundwater:
        add_groundwater_wells(map_object, layers["groundwater_wells"])

    marker_color = "red" if not QATAR_POLYGON.contains(Point(lon, lat)) else "green"
    marker_text = "Selected point - water/offshore" if marker_color == "red" else selected_marker_tooltip(suitability, climate)
    popup_html = selected_marker_popup_html(suitability, climate, selected_report)
    folium.Marker(
        [lat, lon],
        tooltip=folium.Tooltip(marker_text, sticky=True),
        popup=folium.Popup(folium.Html(popup_html, script=False), max_width=380),
        icon=folium.Icon(color=marker_color, icon="leaf"),
    ).add_to(map_object)
    folium.LayerControl(collapsed=True).add_to(map_object)
    map_object.fit_bounds([[24.55, 50.72], [26.18, 51.66]])
    return map_object


def format_distance(distance_m: float) -> str:
    if math.isinf(distance_m):
        return "Unavailable"
    if distance_m >= 1000:
        return f"{distance_m / 1000:,.1f} km"
    return f"{distance_m:,.0f} m"


def money(value: float) -> str:
    return f"QAR {value:,.0f}"


def years(value: float) -> str:
    if math.isinf(value) or value > 100:
        return ">100 years"
    return f"{value:.1f} years"


def score_bar_dataframe(labels: list[str], scores: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"Score": scores}, index=labels)


@st.cache_data(show_spinner=False, ttl=900)
def all_combinations(
    lat: float,
    lon: float,
    area_m2: int,
    transmissivity: float,
    recycle: bool,
    include_microclimate: bool,
    _landuse: gpd.GeoDataFrame,
) -> pd.DataFrame:
    rows = []
    for crop in CROP_DATABASE.values():
        for tech in GREENHOUSE_TECHS.values():
            rows.append(analyze_location(lat, lon, crop, tech, area_m2, transmissivity, recycle, _landuse, include_microclimate=include_microclimate))
    return pd.DataFrame(rows)


def build_pdf_report(site: dict, report_df: pd.DataFrame) -> Optional[bytes]:
    if not REPORTLAB_AVAILABLE:
        return None
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    elements = [
        Paragraph("Qatar Greenhouse Atlas - Site Report", styles["Title"]),
        Paragraph(f"Coordinates: {site['lat']:.4f} N, {site['lon']:.4f} E", styles["Normal"]),
        Paragraph(f"Land use: {site['landuse']} | Suitability: {site['score']}/100", styles["Normal"]),
        Spacer(1, 12),
    ]
    table_df = report_df.head(10)[["crop", "technology", "yield_tons", "total_water_m3", "total_energy_mwh", "net_profit_qar", "payback_years"]].copy()
    table_df.columns = ["Crop", "Technology", "Yield t", "Water m3", "Energy MWh", "Profit QAR", "Payback"]
    data = [list(table_df.columns)] + table_df.round(2).astype(str).values.tolist()
    table = Table(data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e7eb")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#9ca3af")),
                ("FONT", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    elements.append(table)
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("Planning-grade model. Replace synthetic land-use and infrastructure layers with official GeoJSON layers for regulatory use.", styles["Italic"]))
    doc.build(elements)
    return buffer.getvalue()


st.set_page_config(page_title="Qatar Greenhouse Atlas", page_icon="🌱", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 0.65rem; padding-bottom: 1.4rem; max-width: 1500px;}
    div[data-testid="stMetric"] {background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 0.58rem;}
    div[data-testid="stMetric"] label {font-size: 0.78rem;}
    div[data-testid="stMetricValue"] {font-size: 1.05rem;}
    .small-note {font-size: 0.86rem; color: #4b5563;}
    .leaflet-tooltip {white-space: normal !important; max-width: 240px !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

power_lines = load_vector_layer("qatar_kahramaa_lines.geojson", "power")
roads = load_vector_layer("qatar_ashghal_roads.geojson", "roads")
landuse = load_vector_layer("qatar_landuse.geojson", "landuse")
groundwater_wells = load_groundwater_wells()
allowed_landuse_gdf, excluded_landuse_gdf = split_landuse(landuse)
layers = {
    "power": power_lines,
    "roads": roads,
    "groundwater_wells": groundwater_wells,
    "landuse": landuse,
    "allowed_landuse": allowed_landuse_gdf,
    "excluded_landuse": excluded_landuse_gdf,
}

if "selected_lat" not in st.session_state:
    st.session_state.selected_lat = 25.3548
    st.session_state.selected_lon = 51.1839
if "tour_visible" not in st.session_state:
    st.session_state.tour_visible = True
if "tour_step" not in st.session_state:
    st.session_state.tour_step = 0

st.title("Qatar Greenhouse Atlas")
st.caption("Land, water, land-use, crop-technology optimisation, and greenhouse investment screening.")

TOUR_STEPS = [
    (
        "1. Start With The Map",
        "Click any point inside Qatar. The green marker moves to that location and the selected-site panel updates immediately.",
    ),
    (
        "2. Read The Selected-Site Panel",
        "The right panel shows suitability, land use, climate, road/grid distance, groundwater score, and the score breakdown without scrolling.",
    ),
    (
        "3. Use The Marker",
        "Hover over the green marker for a compact summary. Click it for the full popup with groundwater, yield, water, energy, and profit.",
    ),
    (
        "4. Turn Layers On Only When Needed",
        "Use the sidebar to toggle land use, infrastructure, groundwater wells, and the heatmap. Keep the heatmap off for the fastest point analysis.",
    ),
    (
        "5. Compare Or Optimise",
        "Switch views at the top to compare crop-technology options or run investment optimisation for the selected point.",
    ),
]

if st.session_state.tour_visible:
    step_title, step_body = TOUR_STEPS[st.session_state.tour_step]
    with st.container(border=True):
        tour_cols = st.columns([0.14, 0.56, 0.10, 0.10, 0.10], vertical_alignment="center")
        tour_cols[0].markdown(f"**Tour {st.session_state.tour_step + 1}/{len(TOUR_STEPS)}**")
        tour_cols[1].markdown(f"**{step_title}**  \n{step_body}")
        if tour_cols[2].button("Back", disabled=st.session_state.tour_step == 0, use_container_width=True):
            st.session_state.tour_step = max(0, st.session_state.tour_step - 1)
            st.rerun()
        if tour_cols[3].button("Next", disabled=st.session_state.tour_step == len(TOUR_STEPS) - 1, use_container_width=True):
            st.session_state.tour_step = min(len(TOUR_STEPS) - 1, st.session_state.tour_step + 1)
            st.rerun()
        if tour_cols[4].button("Skip", use_container_width=True):
            st.session_state.tour_visible = False
            st.rerun()
else:
    if st.button("Show quick tour"):
        st.session_state.tour_visible = True
        st.session_state.tour_step = 0
        st.rerun()

with st.sidebar:
    st.header("Suitability Weights")
    w_climate = st.slider("Low-humidity microclimate", 0.0, 1.0, 0.26, 0.01)
    w_grid = st.slider("Power grid proximity", 0.0, 1.0, 0.16, 0.01)
    w_logistics = st.slider("Highway logistics", 0.0, 1.0, 0.12, 0.01)
    w_groundwater = st.slider("Groundwater source suitability", 0.0, 1.0, 0.14, 0.01)
    w_landuse = st.slider("Permitted land use", 0.0, 1.0, 0.32, 0.01)
    w_constraints = st.slider("Buffer from exclusions", 0.0, 1.0, 0.10, 0.01)
    total_weight = max(w_climate + w_grid + w_logistics + w_groundwater + w_landuse + w_constraints, 0.01)
    weights = {
        "climate": w_climate / total_weight,
        "grid": w_grid / total_weight,
        "logistics": w_logistics / total_weight,
        "groundwater": w_groundwater / total_weight,
        "landuse": w_landuse / total_weight,
        "constraints": w_constraints / total_weight,
    }

    st.header("Production Assumptions")
    area_m2 = st.number_input("Greenhouse area (m²)", min_value=500, max_value=250_000, value=5_000, step=500)
    transmissivity = st.slider("Cover transmissivity", 0.45, 0.85, 0.65, 0.01)
    recycle = st.toggle("Closed-loop hydroponic drainage", value=True)

    st.header("Performance")
    fast_mode = st.toggle("Fast calculations", value=True, help="Uses a lightweight control-aware microclimate estimate. Turn off for a more detailed estimate that responds more strongly to cover, area, crop, and cooling technology.")

    st.header("Map Layers")
    show_heatmap = st.toggle("Suitability heatmap", value=False, help="The heatmap is cached, but drawing it is still heavier than point-only analysis.")
    heatmap_detail = st.selectbox("Heatmap detail", ["Fast", "Balanced", "Detailed"], index=0, disabled=not show_heatmap)
    heatmap_resolution = {"Fast": 12, "Balanced": 16, "Detailed": 22}[heatmap_detail]
    show_landuse = st.toggle("Land-use layer", value=True)
    show_infra = st.toggle("Infrastructure layers", value=True)
    show_groundwater = st.toggle("Groundwater wells", value=True)

active_view = st.radio(
    "View",
    ["Suitability Map", "Crop-Tech Comparison", "Investment & Optimisation", "Model Notes"],
    horizontal=True,
    label_visibility="collapsed",
)

lat = float(st.session_state.selected_lat)
lon = float(st.session_state.selected_lon)
suitability = calculate_suitability(lat, lon, weights, layers)
climate = interpolate_climate(lat, lon)

if active_view == "Suitability Map":
    default_report = analyze_location(
        lat,
        lon,
        CROP_DATABASE["Tomato - truss/cherry"],
        GREENHOUSE_TECHS["Fan-pad evaporative"],
        int(area_m2),
        transmissivity,
        recycle,
        landuse,
        include_microclimate=not fast_mode,
    )
    map_col, metric_col = st.columns([2.15, 0.85], gap="medium")
    with map_col:
        st.subheader("National Feasibility Map")
        map_object = build_map(
            lat,
            lon,
            weights,
            layers,
            show_heatmap,
            show_infra,
            show_landuse,
            show_groundwater,
            heatmap_resolution,
            suitability=suitability,
            climate=climate,
            selected_report=default_report,
        )
        map_data = st_folium(map_object, height=760, use_container_width=True)
        if map_data and map_data.get("last_clicked"):
            st.session_state.selected_lat = map_data["last_clicked"]["lat"]
            st.session_state.selected_lon = map_data["last_clicked"]["lng"]
            st.rerun()

    with metric_col:
        st.subheader("Selected Site")
        if suitability["is_excluded"]:
            st.error(f"{suitability['status']}. No greenhouse production or investment analysis is valid here.")
        else:
            st.success(suitability["status"])
        st.metric("Suitability index", f"{suitability['score']}/100")
        st.metric("Coordinates", f"{lat:.4f} N, {lon:.4f} E")
        st.metric("Land use", suitability["landuse"], suitability.get("landuse_name", ""))
        if np.isfinite(climate["temp_c"]):
            st.metric("Summer climate", f"{climate['temp_c']} °C / {climate['rh_pct']}% RH", f"{climate['ghi_w_m2']:.0f} W/m² GHI")
        else:
            st.metric("Summer climate", "Not calculated", "water/offshore")
        st.divider()
        c1, c2 = st.columns(2)
        c1.metric("Grid distance", format_distance(float(suitability["grid_distance_m"])))
        c2.metric("Road distance", format_distance(float(suitability["road_distance_m"])))
        c1.metric("Groundwater distance", format_distance(float(suitability["groundwater_distance_m"])))
        c2.metric("Groundwater score", f"{suitability['groundwater_score']:.0f}/100")
        c1.metric("Land-use score", f"{suitability['landuse_score']:.0f}/100")
        c2.metric("Exclusion buffer", format_distance(float(suitability["excluded_distance_m"])))
        st.caption(f"Nearest groundwater feature: {suitability.get('nearest_groundwater', 'Not available')}")
        st.caption(f"Groundwater evidence tier: {suitability.get('groundwater_source_tier', 'Not available')}")

        st.markdown("**Score Breakdown**")
        score_compact_df = score_bar_dataframe(
            ["Climate", "Grid", "Road", "Groundwater", "Land use", "Buffer"],
            [
                suitability["climate_score"],
                suitability["grid_score"],
                suitability["road_score"],
                suitability["groundwater_score"],
                suitability["landuse_score"],
                suitability["constraint_score"],
            ],
        )
        st.bar_chart(score_compact_df, height=185, horizontal=True, use_container_width=True)

        with st.expander("Groundwater score details", expanded=True):
            gw_compact_df = score_bar_dataframe(
                ["Distance", "TDS", "Level", "Yield", "Permit", "Capacity", "Stress"],
                [
                    suitability["groundwater_distance_score"],
                    suitability["groundwater_salinity_score"],
                    suitability["groundwater_level_score"],
                    suitability["groundwater_yield_score"],
                    suitability["groundwater_permit_score"],
                    suitability["groundwater_capacity_score"],
                    suitability["groundwater_stress_score"],
                ],
            )
            st.bar_chart(gw_compact_df, height=205, horizontal=True, use_container_width=True)

        with st.expander("Groundwater quality, quantity and aquifer stress", expanded=False):
            st.write(f"**Aquifer zone:** {suitability.get('groundwater_aquifer_zone', 'Not available')}")
            gw_cols = st.columns(2)
            gw_cols[0].metric("TDS / salinity proxy", "N/A" if not np.isfinite(suitability["groundwater_tds_mg_l"]) else f"{suitability['groundwater_tds_mg_l']:,.0f} mg/L")
            gw_cols[1].metric("Water level", "N/A" if not np.isfinite(suitability["groundwater_level_m_bgl"]) else f"{suitability['groundwater_level_m_bgl']:.0f} m bgl")
            gw_cols[0].metric("Well yield proxy", "N/A" if not np.isfinite(suitability["groundwater_well_yield_m3_day"]) else f"{suitability['groundwater_well_yield_m3_day']:,.0f} m³/day")
            gw_cols[1].metric("Permit allocation", "Missing" if not np.isfinite(suitability["groundwater_permitted_abstraction_m3_year"]) else f"{suitability['groundwater_permitted_abstraction_m3_year']:,.0f} m³/yr")
            gw_cols[0].metric("Transmissivity", "N/A" if not np.isfinite(suitability["groundwater_transmissivity_m2_day"]) else f"{suitability['groundwater_transmissivity_m2_day']:,.0f} m²/day")
            gw_cols[1].metric("Aquifer stress", "N/A" if not np.isfinite(suitability["groundwater_aquifer_stress_ratio"]) else f"{suitability['groundwater_aquifer_stress_ratio']:.1f}x safe yield")
            st.progress(float(max(0.0, min(1.0, suitability["groundwater_confidence"]))))
            st.caption(
                f"Confidence {suitability['groundwater_confidence']:.0%}; exact field completeness "
                f"{suitability['groundwater_data_completeness']:.0%}. {GW_SOURCE_SUMMARY}"
            )

        with st.expander("Site report: tomato + fan-pad", expanded=not suitability["is_excluded"]):
            r1, r2 = st.columns(2)
            r1.metric("Annual yield", f"{default_report['yield_tons']:,.1f} t")
            r2.metric("Total water", f"{default_report['total_water_m3']:,.0f} m³/yr")
            r1.metric("Total energy", f"{default_report['total_energy_mwh']:,.0f} MWh/yr")
            r2.metric("Peak cooling", f"{default_report['peak_cooling_kw']:,.0f} kW")
            r1.metric("Indoor temp", "N/A" if not np.isfinite(default_report["internal_temperature_c"]) else f"{default_report['internal_temperature_c']:.1f} °C")
            r2.metric("ACH", f"{default_report['air_changes_per_hour']:.0f}/h")
            st.metric("Capital investment", money(default_report["capital_qar"]))
            st.metric("Net annual profit", money(default_report["net_profit_qar"]))
            st.metric("Payback", years(default_report["payback_years"]), f"ROI {default_report['roi_percent']:.1f}%")

elif active_view == "Crop-Tech Comparison":
    st.subheader("Compare Crop and Greenhouse Technology Packages")
    pair_cols = st.columns(3)
    selected_reports = []
    for idx, col in enumerate(pair_cols):
        with col:
            crop_name = st.selectbox(f"Crop {idx + 1}", list(CROP_DATABASE.keys()), index=min(idx, len(CROP_DATABASE) - 1), key=f"crop_{idx}")
            tech_name = st.selectbox(f"Technology {idx + 1}", list(GREENHOUSE_TECHS.keys()), index=min(idx + 1, len(GREENHOUSE_TECHS) - 1), key=f"tech_{idx}")
            selected_reports.append(
                analyze_location(
                    lat,
                    lon,
                    CROP_DATABASE[crop_name],
                    GREENHOUSE_TECHS[tech_name],
                    int(area_m2),
                    transmissivity,
                    recycle,
                    landuse,
                    include_microclimate=not fast_mode,
                )
            )

    comparison_df = pd.DataFrame(selected_reports)
    display_cols = [
        "feasible",
        "landuse",
        "crop",
        "technology",
        "yield_tons",
        "total_water_m3",
        "total_energy_mwh",
        "internal_temperature_c",
        "air_changes_per_hour",
        "capital_qar",
        "net_profit_qar",
        "payback_years",
        "roi_percent",
    ]
    st.dataframe(comparison_df[display_cols], hide_index=True, width="stretch")
    st.download_button(
        "Download comparison CSV",
        comparison_df.to_csv(index=False).encode("utf-8"),
        "greenhouse_crop_technology_comparison.csv",
        "text/csv",
    )

elif active_view == "Investment & Optimisation":
    st.subheader("Optimisation for Current Location")
    report_df = all_combinations(lat, lon, int(area_m2), transmissivity, recycle, not fast_mode, landuse)
    objective = st.selectbox("Optimisation objective", ["Maximise net profit", "Minimise water use", "Minimise energy use", "Fastest payback", "Highest ROI"])
    feasible_only = st.toggle("Show feasible land-use sites only", value=True)
    opt_df = report_df.copy()
    if feasible_only:
        opt_df = opt_df[opt_df["feasible"]]

    if opt_df.empty:
        st.error("This selected point is on excluded land use. Choose an agricultural or open-land location to run feasible optimisation.")
    else:
        if objective == "Maximise net profit":
            opt_df = opt_df.sort_values("net_profit_qar", ascending=False)
        elif objective == "Minimise water use":
            opt_df = opt_df.sort_values("total_water_m3", ascending=True)
        elif objective == "Minimise energy use":
            opt_df = opt_df.sort_values("total_energy_mwh", ascending=True)
        elif objective == "Fastest payback":
            opt_df = opt_df.sort_values("payback_years", ascending=True)
        else:
            opt_df = opt_df.sort_values("roi_percent", ascending=False)

        top = opt_df.head(8)
        st.dataframe(
            top[["crop", "technology", "yield_tons", "total_water_m3", "total_energy_mwh", "internal_temperature_c", "air_changes_per_hour", "capital_qar", "net_profit_qar", "payback_years", "roi_percent"]],
            hide_index=True,
            width="stretch",
        )
        best = top.iloc[0]
        st.success(f"Best ranked option: {best['crop']} using {best['technology']} | Net profit {money(best['net_profit_qar'])} | Payback {years(best['payback_years'])}")

    site_payload = {"lat": lat, "lon": lon, "landuse": suitability["landuse"], "score": suitability["score"]}
    st.download_button("Download full optimisation CSV", report_df.to_csv(index=False).encode("utf-8"), "qatar_greenhouse_full_optimisation.csv", "text/csv")
    pdf_bytes = build_pdf_report(site_payload, report_df.sort_values("net_profit_qar", ascending=False))
    if pdf_bytes:
        st.download_button("Download PDF site report", pdf_bytes, "qatar_greenhouse_site_report.pdf", "application/pdf")
    else:
        st.info("PDF export needs the optional reportlab package. CSV export is available now.")

else:
    st.subheader("Scientific and GIS Notes")
    st.markdown(
        f"""
        **Land-use rule:** residential, industrial, protected, flood-prone, and urban expansion polygons are hard exclusions.
        Suitability is forced to 0 on excluded land, even if climate or infrastructure scores are strong.

        **Water rule:** clicks outside the Qatar land mask are classified as `Water/offshore`. The model returns no yield,
        no water demand, no energy demand, and no investment ranking for these points.

        **Replaceable GIS layers:** add official files to `{DATA_DIR}`:
        `qatar_landuse.geojson`, `qatar_kahramaa_lines.geojson`, and `qatar_ashghal_roads.geojson`.
        A land-use file should include `landuse` and preferably `greenhouse_ok` fields.

        **Climate model:** summer temperature, relative humidity, and GHI are interpolated with inverse-distance weighting
        from five Qatar reference stations. The values are planning-grade defaults and should be replaced with measured
        gridded climate data for engineering design.

        **Water model:** irrigation is estimated from a FAO-56 style crop coefficient approach using simplified Hargreaves ET0.

        **Groundwater model:** groundwater is now a composite source-suitability score, not a distance-only layer.
        The score combines distance to the nearest groundwater feature, TDS/salinity, groundwater level, well-yield or
        quantity proxy, permitted abstraction if available, transmissivity, storativity, and aquifer stress relative to
        safe yield. Exact fields in `qatar_groundwater_wells.geojson` override defaults when present. If exact fields
        are missing, the Atlas uses source-tiered national screening proxies from public/peer-reviewed Qatar groundwater
        sources and applies a confidence/completeness penalty.

        **Cooling and microclimate model:** peak cooling combines solar gain, sensible heat, and a humidity penalty.
        A control-aware greenhouse microclimate engine estimates indoor temperature, RH, ventilation rate,
        and air changes per hour using the selected cover transmissivity, area, crop sensitivity, and cooling technology.
        Passive, evaporative, hybrid, mechanical, and fully controlled systems are bounded so active cooling remains near
        the crop setpoint while higher transmissivity and larger houses still increase heat burden. Technology packages
        then translate load into cooling water, electrical demand, capital cost, OPEX, revenue, payback, and ROI.

        **Important:** this is a screening atlas. It is not a substitute for official zoning approval, utility connection
        studies, parcel ownership checks, soil/geotechnical assessment, or detailed HVAC design.
        """
    )
