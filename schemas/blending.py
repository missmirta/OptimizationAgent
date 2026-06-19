"""
Pydantic schema for the Blending problem (e.g. Whiskas cat food).
Used for LLM extraction and completeness checking.
"""

from pydantic import BaseModel
from typing import Optional, Literal


class NutrientRequirement(BaseModel):
    operator: Literal[">=", "<=", "=="]
    value: float


class BlendingProblemInput(BaseModel):
    """
    Partial or complete blending problem — as extracted from user text.
    Fields are Optional so partial extraction is valid.

    nutrients layout: nutrients[nutrient_name][ingredient_name] = fraction
    e.g. nutrients["protein"]["chicken"] = 0.1  (10% protein by weight)
    """
    ingredients: Optional[list[str]] = None
    costs: Optional[dict[str, float]] = None          # cost per gram/unit
    nutrients: Optional[dict[str, dict[str, float]]] = None
    requirements: Optional[dict[str, NutrientRequirement]] = None
    total: Optional[float] = None                     # total batch size (e.g. 100g)

    def missing_fields(self) -> list[str]:
        missing = []
        if not self.ingredients:
            missing.append("ingredients")
        if not self.costs:
            missing.append("ingredient costs")
        if not self.nutrients:
            missing.append("nutritional content per ingredient")
        if not self.requirements:
            missing.append("nutritional requirements / constraints")
        if self.total is None:
            missing.append("total batch quantity")
        # Cross-validation: every ingredient must appear in every nutrient row
        if self.ingredients and self.nutrients and self.requirements:
            for ing in self.ingredients:
                for nutrient in self.requirements:
                    if nutrient not in self.nutrients or ing not in self.nutrients.get(nutrient, {}):
                        missing.append(f"'{nutrient}' value for ingredient '{ing}'")
        return missing

    def is_complete(self) -> bool:
        if self.missing_fields():
            return False
        return True

    def follow_up_questions(self) -> list[str]:
        questions = []

        if not self.ingredients:
            questions.append(
                "What ingredients are available for blending? "
                "(e.g. 'beef, chicken, gel')"
            )
        if self.ingredients and not self.costs:
            questions.append(
                f"What is the cost per unit for each ingredient? "
                f"(e.g. 'beef: $0.013/g, chicken: $0.008/g')"
            )
        if not self.nutrients:
            questions.append(
                "What is the nutritional content (protein, fat, fibre, salt, etc.) "
                "for each ingredient, as a fraction or percentage?"
            )
        if not self.requirements:
            questions.append(
                "What are the nutritional requirements for the blend? "
                "(e.g. 'at least 8g protein, at most 6g fat per 100g')"
            )
        if self.total is None:
            questions.append(
                "What is the total quantity of the blend? "
                "(e.g. '100 grams per can')"
            )
        return questions
