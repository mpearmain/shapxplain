"""
Main SHAP-LLM Explainer module for SHAPXplain.

This module provides the core explainer class that combines SHAP values with
LLM-generated explanations. It handles the processing of SHAP values, interaction
with the LLM, and generation of human-readable explanations.

The explainer supports both single predictions and batch processing, with
optional async capabilities for improved performance.
"""

from typing import List, Dict, Optional, Union, Any, Tuple
from collections import Counter
from functools import lru_cache
import asyncio
import time
import numpy as np
import pandas as pd
import json
from pydantic_ai import Agent

from shapxplain.schemas import (
    SHAPFeatureContribution,
    SHAPExplanationRequest,
    SHAPExplanationResponse,
    SHAPBatchExplanationResponse,
    ContributionDirection,
    SignificanceLevel,
)
from shapxplain.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    generate_explanation_prompt,
    generate_batch_insight_prompt,
)
from shapxplain.utils import logger


class ShapLLMExplainer:
    """
    Combines SHAP value computation with LLM-based interpretation to provide
    human-readable explanations of model predictions.

    Attributes:
        model (Any): The trained machine learning model to explain.
        llm_agent (Agent): Pydantic-AI agent for LLM interaction.
        feature_names (Optional[List[str]]): List of feature names.
        significance_threshold (float): Threshold for determining feature importance.
        max_retries (int): Maximum number of retries for LLM queries.
        retry_delay (float): Base delay between retries (seconds).
    """

    def __init__(
        self,
        model: Any,
        llm_agent: Optional[Agent] = None,
        feature_names: Optional[List[str]] = None,
        significance_threshold: float = 0.1,
        system_prompt: Optional[str] = None,
        cache_size: int = 1000,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize the SHAP-LLM explainer.

        Args:
            model (Any): The trained machine learning model to explain.
            llm_agent (Optional[Agent]): LLM agent for querying explanations.
            feature_names (Optional[List[str]]): List of feature names.
            significance_threshold (float): Threshold for determining feature importance.
            system_prompt (Optional[str]): Custom system prompt for the LLM.
            cache_size (int): Size of the LRU cache for LLM queries.
            max_retries (int): Maximum number of retries for failed LLM queries.
            retry_delay (float): Base delay between retries in seconds.

        Raises:
            ValueError: If input parameters are invalid.
        """
        if model is None:
            raise ValueError("Model cannot be None")
        if feature_names is not None and not all(
            isinstance(name, str) for name in feature_names
        ):
            raise ValueError("All feature names must be strings")
        if not isinstance(significance_threshold, (int, float)):
            raise ValueError("significance_threshold must be numeric")
        if significance_threshold <= 0:
            raise ValueError("significance_threshold must be positive")
        if significance_threshold > 1:
            raise ValueError("significance_threshold should typically be <= 1")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if retry_delay <= 0:
            raise ValueError("retry_delay must be positive")

        self.model = model
        self.feature_names = feature_names
        self.significance_threshold = significance_threshold
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._query_llm_impl = lru_cache(maxsize=cache_size)(self._query_llm_impl)

        logger.info(
            f"Initializing ShapLLMExplainer with model type: {model.__class__.__name__}"
        )
        logger.debug(
            f"Significance threshold: {significance_threshold}, Max retries: {max_retries}"
        )

        # Initialize LLM agent
        if llm_agent:
            self.llm_agent = llm_agent
            logger.info(f"Using provided LLM agent: {type(llm_agent).__name__}")
        else:
            model_name = "openai:gpt-4o-mini"  # Default model
            logger.info(
                f"No LLM agent provided, creating default agent with model: {model_name}"
            )
            self.llm_agent = Agent(
                model=model_name, system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT
            )

    def _process_shap_values(
        self,
        shap_values: Union[np.ndarray, List],
        data_point: Union[np.ndarray, pd.Series],
    ) -> List[SHAPFeatureContribution]:
        """
        Process SHAP values into structured feature contributions.

        Args:
            shap_values: SHAP values for a single prediction.
            data_point: Original feature values for the prediction.

        Returns:
            List[SHAPFeatureContribution]: Structured feature contributions with significance and direction.

        Raises:
            ValueError: If input dimensions don't match.
        """
        if len(shap_values) != len(data_point):
            raise ValueError(
                f"Length mismatch: shap_values ({len(shap_values)}) != "
                f"data_point ({len(data_point)})"
            )
        if self.feature_names and len(self.feature_names) != len(shap_values):
            raise ValueError(
                f"Feature names length ({len(self.feature_names)}) doesn't match "
                f"shap_values length ({len(shap_values)})"
            )

        if isinstance(shap_values, list):
            shap_values = np.array(shap_values)

        max_abs_shap = np.max(np.abs(shap_values))
        normalized_shap = (
            shap_values / max_abs_shap if max_abs_shap > 0 else shap_values
        )

        contributions = []
        for idx, (shap_val, orig_val) in enumerate(zip(normalized_shap, data_point)):
            feature_name = (
                self.feature_names[idx] if self.feature_names else f"feature_{idx}"
            )
            direction = self._determine_direction(shap_val)
            significance = self._determine_significance(shap_val)

            contributions.append(
                SHAPFeatureContribution(
                    feature_name=feature_name,
                    shap_value=float(shap_val),
                    original_value=orig_val,
                    contribution_direction=direction,
                    significance=significance,
                )
            )

        return sorted(contributions, key=lambda x: abs(x.shap_value), reverse=True)

    @staticmethod
    def _determine_direction(shap_value: float) -> ContributionDirection:
        """
        Determine the direction of a feature's contribution.

        Args:
            shap_value: Normalized SHAP value for a feature.

        Returns:
            ContributionDirection: Direction of the feature's impact.
        """
        if abs(shap_value) < 1e-10:
            return ContributionDirection.NEUTRAL
        return (
            ContributionDirection.INCREASE
            if shap_value > 0
            else ContributionDirection.DECREASE
        )

    def _determine_significance(self, shap_value: float) -> SignificanceLevel:
        """
        Determine the significance level of a feature's contribution.

        Args:
            shap_value: Normalized SHAP value for a feature.

        Returns:
            SignificanceLevel: Significance level (HIGH, MEDIUM, or LOW).
        """
        abs_val = abs(shap_value)
        if abs_val >= self.significance_threshold * 2:
            return SignificanceLevel.HIGH
        elif abs_val >= self.significance_threshold:
            return SignificanceLevel.MEDIUM
        return SignificanceLevel.LOW

    @staticmethod
    def _clean_json_response(text: str) -> str:
        """Clean JSON response by removing Markdown code blocks if present.

        Args:
            text: Raw response text from LLM

        Returns:
            str: Cleaned JSON string
        """
        # Handle Markdown code blocks with language identifier
        if text.startswith("`") and text.endswith("`"):
            lines = text.split("\n")
            if len(lines) >= 2:
                # Skip first line (which may be `json) and last line (which is `)
                start_idx = 1
                if lines[0].startswith("```") and len(lines[0]) > 3:
                    # We have a language identifier like ```json
                    start_idx = 1
                text = "\n".join(lines[start_idx:-1])
        return text.strip()

    def _query_llm(self, prompt: str) -> dict:
        """
        Query the LLM agent with retries and ensure it returns a valid dictionary response.

        This method wraps the cached implementation and adds retry logic.

        Args:
            prompt (str): The prompt to send to the LLM.

        Returns:
            dict: Parsed JSON response from the LLM.

        Raises:
            RuntimeError: If the query fails or response is invalid after retries.
        """
        for attempt in range(self.max_retries + 1):
            try:
                return self._query_llm_impl(prompt)
            except (json.JSONDecodeError, RuntimeError) as e:
                if attempt == self.max_retries:
                    raise RuntimeError(
                        f"Failed to query LLM after {self.max_retries} attempts: {e}"
                    )
                # Exponential backoff: delay = base_delay * 2^attempt
                delay = self.retry_delay * (2**attempt)
                time.sleep(delay)

    def _query_llm_impl(self, prompt: str) -> dict:
        """
        Cached implementation for LLM querying.

        Args:
            prompt (str): The prompt to send to the LLM.

        Returns:
            dict: Parsed JSON response from the LLM.

        Raises:
            RuntimeError: If the query fails or response is invalid.
        """
        result = None
        try:
            logger.debug(
                f"Sending query to LLM with prompt length: {len(prompt)} chars"
            )
            result = self.llm_agent.run_sync(prompt)
            response_text = result.data
            logger.debug(
                f"Received LLM response with length: {len(response_text)} chars"
            )
            cleaned_text = self._clean_json_response(response_text)
            return json.loads(cleaned_text)  # Expecting JSON
        except json.JSONDecodeError:
            response_text = result.data if result else "<no response>"
            logger.error(f"LLM response is not valid JSON: {response_text[:100]}...")
            raise RuntimeError(f"LLM response is not valid JSON: {response_text}")
        except Exception as e:
            logger.error(f"Failed to query the LLM: {str(e)}")
            raise RuntimeError(f"Failed to query the LLM: {e}")

    async def _query_llm_async(self, prompt: str) -> dict:
        """
        Asynchronous version of LLM querying with retries.

        Args:
            prompt (str): The prompt to send to the LLM.

        Returns:
            dict: Parsed JSON response from the LLM.

        Raises:
            RuntimeError: If the query fails or response is invalid after retries.
        """
        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(
                    f"Sending async query to LLM, attempt {attempt+1}/{self.max_retries+1}"
                )
                result = await self.llm_agent.run(prompt)
                response_text = result.data
                logger.debug(
                    f"Received async LLM response with length: {len(response_text)} chars"
                )
                cleaned_text = self._clean_json_response(response_text)
                return json.loads(cleaned_text)  # Expecting JSON
            except json.JSONDecodeError:
                if attempt == self.max_retries:
                    logger.error(
                        f"LLM response is not valid JSON after {self.max_retries} attempts"
                    )
                    raise RuntimeError(
                        f"LLM response is not valid JSON after {self.max_retries} attempts"
                    )
                # Exponential backoff
                delay = self.retry_delay * (2**attempt)
                logger.warning(
                    f"JSON decode error, retrying in {delay:.2f}s (attempt {attempt+1}/{self.max_retries})"
                )
                await asyncio.sleep(delay)
            except Exception as e:
                if attempt == self.max_retries:
                    logger.error(
                        f"Failed to query the LLM after {self.max_retries} attempts: {str(e)}"
                    )
                    raise RuntimeError(
                        f"Failed to query the LLM after {self.max_retries} attempts: {e}"
                    )
                delay = self.retry_delay * (2**attempt)
                logger.warning(
                    f"LLM query error: {str(e)}, retrying in {delay:.2f}s (attempt {attempt+1}/{self.max_retries})"
                )
                await asyncio.sleep(delay)

    def _create_explanation_request(
        self,
        shap_values: Union[np.ndarray, List],
        data_point: Union[np.ndarray, pd.Series],
        prediction: float,
        prediction_class: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, SHAPExplanationRequest]:
        """Common setup for both sync and async explain methods."""
        feature_contributions = self._process_shap_values(shap_values, data_point)

        request = SHAPExplanationRequest(
            model_type=self.model.__class__.__name__,
            prediction=float(prediction),
            prediction_class=prediction_class,
            features=feature_contributions,
            context=additional_context or {},
        )

        prompt = generate_explanation_prompt(
            model_type=request.model_type,
            prediction=request.prediction,
            features=request.features,
            prediction_class=request.prediction_class,
            context=request.context,
        )

        return prompt, request

    def explain(
        self,
        shap_values: Union[np.ndarray, List],
        data_point: Union[np.ndarray, pd.Series],
        prediction: float,
        prediction_class: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> SHAPExplanationResponse:
        """
        Generate a detailed explanation for a single prediction.

        Args:
            shap_values: SHAP values for the prediction.
            data_point: Original feature values for the prediction.
            prediction: Model's numerical prediction.
            prediction_class: Optional class label for classification tasks.
            additional_context: Optional context for better explanations.

        Returns:
            SHAPExplanationResponse: Structured explanation including summary and recommendations.

        Raises:
            RuntimeError: If the LLM query fails.
        """
        logger.info(
            f"Generating explanation for prediction: {prediction:.4f}"
            + (f" (class: {prediction_class})" if prediction_class else "")
        )

        prompt, _ = self._create_explanation_request(
            shap_values, data_point, prediction, prediction_class, additional_context
        )

        logger.debug("Querying LLM for explanation")
        response = self._query_llm(prompt)

        logger.info("Explanation generated successfully")
        return SHAPExplanationResponse.model_validate(response)

    async def explain_async(
        self,
        shap_values: Union[np.ndarray, List],
        data_point: Union[np.ndarray, pd.Series],
        prediction: float,
        prediction_class: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> SHAPExplanationResponse:
        """
        Asynchronous version of explain method.

        Args:
            shap_values: SHAP values for the prediction.
            data_point: Original feature values for the prediction.
            prediction: Model's numerical prediction.
            prediction_class: Optional class label for classification tasks.
            additional_context: Optional context for better explanations.

        Returns:
            SHAPExplanationResponse: Structured explanation including summary and recommendations.

        Raises:
            RuntimeError: If the LLM query fails.
        """
        logger.info(
            f"Generating async explanation for prediction: {prediction:.4f}"
            + (f" (class: {prediction_class})" if prediction_class else "")
        )

        prompt, _ = self._create_explanation_request(
            shap_values, data_point, prediction, prediction_class, additional_context
        )

        logger.debug("Querying LLM asynchronously for explanation")
        response = await self._query_llm_async(prompt)

        logger.info("Async explanation generated successfully")
        return SHAPExplanationResponse.model_validate(response)

    def _calculate_confidence_summary(
        self, shap_values_batch: Union[np.ndarray, List]
    ) -> Dict[SignificanceLevel, int]:
        """
        Calculate confidence summary based on SHAP values and significance levels.

        Args:
            shap_values_batch: Batch of SHAP values for predictions.

        Returns:
            Dict[SignificanceLevel, int]: Counts of predictions in each significance level.
        """
        if isinstance(shap_values_batch, list):
            shap_values_batch = np.array(shap_values_batch)

        max_abs_shap = np.max(np.abs(shap_values_batch), axis=1)

        high_mask = max_abs_shap >= self.significance_threshold * 2
        med_mask = (max_abs_shap >= self.significance_threshold) & ~high_mask

        return {
            SignificanceLevel.HIGH: int(np.sum(high_mask)),
            SignificanceLevel.MEDIUM: int(np.sum(med_mask)),
            SignificanceLevel.LOW: int(np.sum(~high_mask & ~med_mask)),
        }

    def explain_batch(
        self,
        shap_values_batch: Union[np.ndarray, List],
        data_points: Union[np.ndarray, List],
        predictions: Union[np.ndarray, List[float]],
        prediction_classes: Optional[List[str]] = None,
        batch_size: int = 5,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> SHAPBatchExplanationResponse:
        """
        Generate explanations for multiple predictions in batches.

        Args:
            shap_values_batch: Batch of SHAP values.
            data_points: Batch of original feature values.
            predictions: Model predictions for each data point.
            prediction_classes: Optional class labels for classification tasks.
            batch_size: Number of explanations to process simultaneously.
            additional_context: Optional context for better explanations.

        Returns:
            SHAPBatchExplanationResponse: Batch response with explanations and insights.

        Raises:
            ValueError: If input dimensions don't match.
            RuntimeError: If LLM query fails.
        """
        self._validate_batch_inputs(
            shap_values_batch, data_points, predictions, prediction_classes
        )

        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        responses = []
        for i in range(0, len(predictions), batch_size):
            batch_end = min(i + batch_size, len(predictions))
            batch_responses = [
                self.explain(
                    shap_values_batch[j],
                    data_points[j],
                    predictions[j],
                    prediction_classes[j] if prediction_classes else None,
                    additional_context,
                )
                for j in range(i, batch_end)
            ]
            responses.extend(batch_responses)

        return self._generate_batch_response(responses, shap_values_batch, predictions)

    async def explain_batch_async(
        self,
        shap_values_batch: Union[np.ndarray, List],
        data_points: Union[np.ndarray, List],
        predictions: Union[np.ndarray, List[float]],
        prediction_classes: Optional[List[str]] = None,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> SHAPBatchExplanationResponse:
        """
        Asynchronously generate explanations for multiple predictions in parallel.

        Args:
            shap_values_batch: Batch of SHAP values.
            data_points: Batch of original feature values.
            predictions: Model predictions for each data point.
            prediction_classes: Optional class labels for classification tasks.
            additional_context: Optional context for better explanations.

        Returns:
            SHAPBatchExplanationResponse: Batch response with explanations and insights.

        Raises:
            ValueError: If input dimensions don't match.
            RuntimeError: If LLM query fails.
        """
        self._validate_batch_inputs(
            shap_values_batch, data_points, predictions, prediction_classes
        )

        # Create tasks for all explanations to run concurrently
        tasks = []
        for i in range(len(predictions)):
            tasks.append(
                self.explain_async(
                    shap_values_batch[i],
                    data_points[i],
                    predictions[i],
                    prediction_classes[i] if prediction_classes else None,
                    additional_context,
                )
            )

        # Run all explanation tasks in parallel
        responses = await asyncio.gather(*tasks)

        return self._generate_batch_response(responses, shap_values_batch, predictions)

    def _validate_batch_inputs(
        self,
        shap_values_batch: Union[np.ndarray, List],
        data_points: Union[np.ndarray, List],
        predictions: Union[np.ndarray, List[float]],
        prediction_classes: Optional[List[str]] = None,
    ) -> None:
        """
        Validate inputs for batch processing.

        Args:
            shap_values_batch: Batch of SHAP values.
            data_points: Batch of original feature values.
            predictions: Model predictions for each data point.
            prediction_classes: Optional class labels for classification tasks.

        Raises:
            ValueError: If input dimensions don't match.
        """
        if len(shap_values_batch) != len(data_points) or len(shap_values_batch) != len(
            predictions
        ):
            raise ValueError("Length mismatch in batch inputs")
        if prediction_classes is not None and len(prediction_classes) != len(
            predictions
        ):
            raise ValueError("Length mismatch in prediction_classes")

    def _generate_batch_response(
        self,
        responses: List[SHAPExplanationResponse],
        shap_values_batch: Union[np.ndarray, List],
        predictions: Union[np.ndarray, List[float]],
    ) -> SHAPBatchExplanationResponse:
        """
        Generate the batch response from individual explanations.

        Args:
            responses: List of individual explanation responses.
            shap_values_batch: Batch of SHAP values for confidence summary.
            predictions: Model predictions for batch insights.

        Returns:
            SHAPBatchExplanationResponse: Batch response with explanations and insights.
        """
        # Batch insights
        common_features = self._identify_common_features(responses)
        confidence_summary = self._calculate_confidence_summary(shap_values_batch)

        confidence_distribution = ", ".join(
            f"{level.value.title()}: {count}"
            for level, count in confidence_summary.items()
        )

        batch_prompt = generate_batch_insight_prompt(
            model_type=self.model.__class__.__name__,
            predictions=predictions,
            common_features=common_features,
            confidence_summary=confidence_summary,
            confidence_distribution=confidence_distribution,
        )

        batch_insights_response = self._query_llm(batch_prompt)

        return SHAPBatchExplanationResponse(
            responses=responses,
            summary_statistics={
                "total_processed": len(responses),
                "confidence_summary": confidence_summary,
                "common_features": common_features,
            },
            batch_insights=batch_insights_response.get("insights", []),
        )

    @staticmethod
    def _identify_common_features(
        responses: List[SHAPExplanationResponse],
    ) -> List[Tuple[str, int]]:
        """
        Identify commonly important features across multiple explanations.

        Args:
            responses: List of explanation responses to analyze.

        Returns:
            List[Tuple[str, int]]: List of feature names and their counts that are consistently important.
        """
        feature_counts = Counter()
        for response in responses:
            for feature in response.features:
                if feature.significance in [
                    SignificanceLevel.HIGH,
                    SignificanceLevel.MEDIUM,
                ]:
                    feature_counts[feature.feature_name] += 1

        # Return features that appear in at least half the responses, with their counts
        threshold = len(responses) // 2
        return [
            (feature, count)
            for feature, count in feature_counts.most_common()
            if count > threshold
        ]
