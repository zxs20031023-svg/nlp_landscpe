from __future__ import annotations

from copy import deepcopy

from .models import AppliedRule, LandscapeBrief, SiteAnalysis


class ComplianceRuleEngine:
    def __init__(self, rules_config: dict, sense_mapping: dict[str, str]) -> None:
        self.rules = rules_config
        self.sense_mapping = sense_mapping

    @staticmethod
    def _contains_any(text: str, keywords: list[str]) -> bool:
        return any(keyword in text for keyword in keywords)

    @staticmethod
    def _append_unique(items: list[str], value: str) -> None:
        if value not in items:
            items.append(value)

    def apply_to_brief(self, brief: LandscapeBrief, user_input: str) -> tuple[LandscapeBrief, list[AppliedRule]]:
        updated = brief.model_copy(deep=True)
        updated.functional_zones = list(updated.functional_zones)
        updated.warnings = list(updated.warnings or [])
        applied_rules: list[AppliedRule] = []

        updated.canopy_closure = round(min(max(updated.canopy_closure, 0.0), 1.0), 2)
        updated.hardscape_ratio = round(min(max(updated.hardscape_ratio, 0.0), 1.0), 2)
        if updated.path_slope_max_percentage < 0:
            updated.path_slope_max_percentage = 0

        access_keywords = self.rules.get("accessible_keywords", [])
        max_accessible_slope = float(self.rules.get("max_accessible_slope", 5.0))
        if self._contains_any(user_input, access_keywords) and updated.path_slope_max_percentage > max_accessible_slope:
            original_value = updated.path_slope_max_percentage
            updated.path_slope_max_percentage = max_accessible_slope
            warning = (
                f"检测到无障碍或适老通行需求，园路最大坡度由 {original_value}% "
                f"自动修正为 {max_accessible_slope}% 以满足安全通行要求。"
            )
            self._append_unique(updated.warnings, warning)
            applied_rules.append(
                AppliedRule(
                    rule_id="accessible-slope",
                    title="无障碍坡度控制",
                    warning=warning,
                    payload={"from": original_value, "to": max_accessible_slope},
                )
            )

        zone_keywords = self.rules.get("required_zone_keywords", {})
        for keyword, zone_name in zone_keywords.items():
            if keyword in user_input:
                self._append_unique(updated.functional_zones, zone_name)

        toxic_plants: dict[str, str] = self.rules.get("toxic_plants", {})
        protective_keywords = self.rules.get("protective_keywords", [])
        if self._contains_any(user_input, protective_keywords):
            for plant_name, replacements in toxic_plants.items():
                if plant_name in user_input:
                    warning = (
                        f"检测到敏感人群活动场景且需求中包含 `{plant_name}`，"
                        f"建议替换为 {replacements} 等无毒安全树种。"
                    )
                    self._append_unique(updated.warnings, warning)
                    applied_rules.append(
                        AppliedRule(
                            rule_id=f"replace-{plant_name}",
                            title="敏感植物替换建议",
                            warning=warning,
                            payload={"plant": plant_name, "replacement": replacements},
                        )
                    )

        wetland_keywords = self.rules.get("wetland_keywords", [])
        if self._contains_any(user_input, wetland_keywords):
            warning = "涉及湿地、水岸或积水场景，建议同步配置生态驳岸、雨水花园和安全防护节点。"
            self._append_unique(updated.warnings, warning)
            applied_rules.append(
                AppliedRule(
                    rule_id="wetland-ecology",
                    title="湿地与水岸生态提醒",
                    warning=warning,
                )
            )
            self._append_unique(updated.functional_zones, "雨水花园/生态缓冲带")

        if "生态" in updated.style_preference and updated.hardscape_ratio > 0.4:
            original_ratio = updated.hardscape_ratio
            updated.hardscape_ratio = 0.4
            warning = (
                f"风格偏好包含生态导向，硬质铺装比例由 {original_ratio} 调整为 0.4，"
                "以保证绿量和雨水渗透空间。"
            )
            self._append_unique(updated.warnings, warning)
            applied_rules.append(
                AppliedRule(
                    rule_id="ecology-hardscape",
                    title="生态场景铺装率修正",
                    warning=warning,
                )
            )

        return updated, applied_rules

    def apply_to_site_analysis(
        self, analysis: SiteAnalysis, site_text: str
    ) -> tuple[SiteAnalysis, list[AppliedRule]]:
        updated = analysis.model_copy(deep=True)
        updated.opportunities = list(updated.opportunities)
        updated.constraints = list(updated.constraints)
        updated.design_suggestions = list(updated.design_suggestions)
        applied_rules: list[AppliedRule] = []

        site_suggestion_rules = self.rules.get("site_suggestion_rules", [])
        for item in site_suggestion_rules:
            if self._contains_any(site_text, item["keywords"]):
                for suggestion in item.get("suggestions", []):
                    self._append_unique(updated.design_suggestions, suggestion)
                for constraint in item.get("constraints", []):
                    self._append_unique(updated.constraints, constraint)
                applied_rules.append(
                    AppliedRule(
                        rule_id=item["id"],
                        title=item["title"],
                        warning=item["summary"],
                        payload=deepcopy(item),
                    )
                )

        return updated, applied_rules
