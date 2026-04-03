from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from landscape_workflow.models import LandscapeBrief, SiteAnalysis
from landscape_workflow.rules import ComplianceRuleEngine


RULES = {
    "accessible_keywords": ["无障碍", "轮椅", "适老"],
    "protective_keywords": ["儿童", "老人", "社区"],
    "wetland_keywords": ["湿地", "积水"],
    "required_zone_keywords": {"儿童": "儿童活动区", "适老": "适老康体区"},
    "toxic_plants": {"夹竹桃": "桂花、木槿"},
    "max_accessible_slope": 5.0,
    "site_suggestion_rules": [
        {
            "id": "waterlogging-site",
            "title": "低洼积水生态排水补充",
            "keywords": ["积水", "低洼"],
            "summary": "需要补充雨洪和排水建议。",
            "suggestions": ["增设雨水花园"],
            "constraints": ["校核暴雨回排"]
        }
    ]
}


class RuleEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = ComplianceRuleEngine(rules_config=RULES, sense_mapping={})

    def test_accessible_slope_is_corrected(self):
        brief = LandscapeBrief(
            project_type="社区口袋公园",
            style_preference="生态适老",
            canopy_closure=0.6,
            path_slope_max_percentage=8.0,
            hardscape_ratio=0.35,
            functional_zones=[],
            warnings=[],
        )
        updated, rules = self.engine.apply_to_brief(
            brief=brief,
            user_input="设计一个适老社区公园，轮椅可以通行。",
        )
        self.assertEqual(updated.path_slope_max_percentage, 5.0)
        self.assertTrue(updated.warnings)
        self.assertTrue(rules)

    def test_toxic_plant_warning_is_added(self):
        brief = LandscapeBrief(
            project_type="社区口袋公园",
            style_preference="自然",
            canopy_closure=0.5,
            path_slope_max_percentage=3.0,
            hardscape_ratio=0.25,
            functional_zones=[],
            warnings=[],
        )
        updated, _ = self.engine.apply_to_brief(
            brief=brief,
            user_input="社区儿童活动区附近种植夹竹桃。",
        )
        self.assertIn("儿童活动区", updated.functional_zones)
        self.assertTrue(any("夹竹桃" in warning for warning in updated.warnings))

    def test_site_analysis_rules_append_suggestions(self):
        analysis = SiteAnalysis(
            location_context="老旧社区内部",
            climate_environment="通风一般",
            topography_features="场地低洼，存在积水",
            opportunities=["需求明确"],
            constraints=[],
            design_suggestions=[],
        )
        updated, rules = self.engine.apply_to_site_analysis(
            analysis=analysis,
            site_text="场地低洼并长期积水。",
        )
        self.assertIn("增设雨水花园", updated.design_suggestions)
        self.assertIn("校核暴雨回排", updated.constraints)
        self.assertTrue(rules)


if __name__ == "__main__":
    unittest.main()
