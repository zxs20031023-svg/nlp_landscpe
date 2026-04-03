from pathlib import Path
import json
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from landscape_workflow.project_recommender import ProjectRecommendationEngine


class RecommendationEngineTests(unittest.TestCase):
    def test_recommend_returns_matching_case(self):
        engine = ProjectRecommendationEngine()
        results = engine.recommend(
            site_text="高密度社区中的口袋公园更新，强调适老、儿童活动和夜间安全。",
            top_k=3,
        )
        self.assertTrue(results)
        self.assertIn("口袋公园", results[0].project_type)

    def test_add_cases_to_library_skips_duplicate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            library_path = Path(temp_dir) / "project_case_library.json"
            library_path.write_text(json.dumps({"projects": []}, ensure_ascii=False), encoding="utf-8")
            engine = ProjectRecommendationEngine(library_path=library_path)

            payload = [
                {
                    "project_id": "online_case_1",
                    "name": "测试在线案例",
                    "city": "在线检索",
                    "project_type": "在线案例",
                    "scene": "社区更新",
                    "summary": "测试用案例",
                    "source_url": "https://example.com/case-1",
                    "source_label": "DuckDuckGo",
                    "library_source": "online_imported",
                    "keywords": ["社区更新"],
                    "target_users": [],
                    "highlights": ["支持导入"],
                    "matching_points": ["社区更新"],
                    "recommendation_reason": "测试导入",
                    "similarity_score": 88.0,
                }
            ]

            imported_count, _ = engine.add_cases_to_library(payload)
            duplicated_count, _ = engine.add_cases_to_library(payload)

            self.assertEqual(imported_count, 1)
            self.assertEqual(duplicated_count, 0)
            saved = json.loads(library_path.read_text(encoding="utf-8"))
            self.assertEqual(len(saved["projects"]), 1)


if __name__ == "__main__":
    unittest.main()
