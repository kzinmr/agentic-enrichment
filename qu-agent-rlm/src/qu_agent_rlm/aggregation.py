from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
from datetime import date
from itertools import product
from typing import Any


ALLOWED_AGGREGATION_FUNCTIONS = {
    "cohort_count",
    "count",
    "count_if",
    "count_where",
    "date_range_count",
    "group_count",
    "nested_group_count",
    "numeric_range_count",
    "ratio",
    "top_k",
}

ALLOWED_KEYWORDS = {"date_granularity", "end", "k", "max_value", "min_value", "precision", "start"}
COMPARISON_OPERATORS = {"==", "!=", ">", ">=", "<", "<="}


@dataclass(frozen=True)
class AggregationEvaluation:
    result: dict[str, Any]
    used_fields: list[str]
    expression: str | None = None


def grouped_count(records: list[dict[str, Any]], group_by: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for record in records:
        for value in field_values(record, group_by):
            counter[str(value)] += 1
    return sorted_counter(counter)


def evaluate_aggregation_expression(
    expression: str,
    records: list[dict[str, Any]],
    fields: dict[str, dict[str, Any]],
) -> AggregationEvaluation:
    used_fields = validate_aggregation_expression(expression, fields)
    parsed = ast.parse(expression, mode="eval")
    namespace = {
        "records": records,
        "cohort_count": cohort_count,
        "count": count_records,
        "count_if": count_if,
        "count_where": count_where,
        "date_range_count": date_range_count,
        "group_count": group_count,
        "nested_group_count": nested_group_count,
        "numeric_range_count": numeric_range_count,
        "ratio": ratio,
        "top_k": top_k,
    }
    result = eval(compile(parsed, "<aggregate_silver>", "eval"), {"__builtins__": {}}, namespace)
    return AggregationEvaluation(
        result=normalize_aggregation_result(result),
        used_fields=used_fields,
        expression=expression,
    )


def validate_aggregation_expression(expression: str, fields: dict[str, dict[str, Any]]) -> list[str]:
    expression = expression.strip()
    if not expression:
        raise ValueError("aggregation expression cannot be empty")
    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid aggregation expression: {exc.msg}") from exc
    validate_aggregation_ast(parsed)
    used_fields = sorted(extract_used_fields(parsed, fields))
    for field_name in used_fields:
        field = fields[field_name]
        if not field.get("search", {}).get("aggregatable", False):
            raise ValueError(f"Aggregation expression referenced non-aggregatable field: {field_name}")
    return used_fields


def validate_aggregation_ast(parsed: ast.AST) -> None:
    for node in ast.walk(parsed):
        if isinstance(node, ast.Expression | ast.Load | ast.Constant | ast.List | ast.Tuple | ast.keyword):
            continue
        if isinstance(node, ast.UAdd | ast.USub):
            continue
        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, ast.UAdd | ast.USub) or not isinstance(node.operand, ast.Constant):
                raise ValueError("Aggregation expressions only allow unary signs on numeric literals")
            continue
        if isinstance(node, ast.Name):
            if node.id != "records" and node.id not in ALLOWED_AGGREGATION_FUNCTIONS:
                raise ValueError(f"Disallowed name in aggregation expression: {node.id}")
            continue
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED_AGGREGATION_FUNCTIONS:
                raise ValueError("Aggregation expression can only call allowlisted functions")
            for keyword in node.keywords:
                if keyword.arg is None or keyword.arg not in ALLOWED_KEYWORDS:
                    raise ValueError(f"Disallowed aggregation keyword: {keyword.arg}")
            continue
        raise ValueError(f"Disallowed syntax in aggregation expression: {type(node).__name__}")


def extract_used_fields(parsed: ast.AST, fields: dict[str, dict[str, Any]]) -> set[str]:
    used: set[str] = set()
    for node in ast.walk(parsed):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        function_name = node.func.id
        if function_name in {"group_count", "top_k", "count_if", "count_where", "numeric_range_count"}:
            field_node = positional_arg(node, 1) or keyword_arg(node, "field")
            used.add(field_from_node(field_node, fields))
        elif function_name == "nested_group_count":
            fields_node = positional_arg(node, 1) or keyword_arg(node, "fields")
            used.update(fields_from_node(fields_node, fields))
    return used


def positional_arg(node: ast.Call, index: int) -> ast.AST | None:
    return node.args[index] if len(node.args) > index else None


def keyword_arg(node: ast.Call, name: str) -> ast.AST | None:
    for keyword in node.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def field_from_node(node: ast.AST | None, fields: dict[str, dict[str, Any]]) -> str:
    if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
        raise ValueError("Aggregation field references must be string literals")
    field_name = node.value
    if field_name not in fields:
        raise ValueError(f"Unknown aggregation field: {field_name}")
    return field_name


def fields_from_node(node: ast.AST | None, fields: dict[str, dict[str, Any]]) -> list[str]:
    if not isinstance(node, ast.List | ast.Tuple):
        raise ValueError("nested_group_count fields must be a literal list of field names")
    return [field_from_node(item, fields) for item in node.elts]


def count_records(records: list[dict[str, Any]]) -> int:
    return len(records)


def group_count(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    return grouped_count(records, field)


def top_k(records: list[dict[str, Any]], field: str, k: int = 5) -> dict[str, int]:
    counter = Counter()
    for record in records:
        counter.update(str(value) for value in field_values(record, field))
    limited = counter.most_common(max(1, int(k)))
    return dict(sorted(limited, key=lambda item: (-item[1], item[0])))


def nested_group_count(records: list[dict[str, Any]], fields: list[str]) -> dict[str, Any]:
    if not fields:
        return {}
    root: dict[str, Any] = {}
    for record in records:
        value_sets = [field_values(record, field) for field in fields]
        if any(not values for values in value_sets):
            continue
        for combination in product(*value_sets):
            current = root
            for value in combination[:-1]:
                current = current.setdefault(str(value), {})
            leaf = str(combination[-1])
            current[leaf] = int(current.get(leaf, 0)) + 1
    return sort_nested_counts(root)


def count_if(records: list[dict[str, Any]], field: str, expected: Any) -> int:
    return sum(1 for record in records if value_matches(record.get("fields", {}).get(field), expected))


def count_where(records: list[dict[str, Any]], field: str, operator: str, value: Any) -> int:
    if operator not in COMPARISON_OPERATORS:
        raise ValueError(f"Unsupported count_where operator: {operator}")
    return sum(1 for record in records if compare_value(record.get("fields", {}).get(field), operator, value))


def numeric_range_count(
    records: list[dict[str, Any]],
    field: str,
    min_value: float | int | None = None,
    max_value: float | int | None = None,
) -> dict[str, Any]:
    matched_ids: list[str] = []
    for record in records:
        number = numeric_value(record.get("fields", {}).get(field))
        if number is None:
            continue
        if min_value is not None and number < float(min_value):
            continue
        if max_value is not None and number > float(max_value):
            continue
        matched_ids.append(str(record.get("call_id")))
    return {
        "count": len(matched_ids),
        "field": field,
        "min_value": min_value,
        "max_value": max_value,
        "record_ids": matched_ids[:20],
    }


def date_range_count(records: list[dict[str, Any]], start: str | None = None, end: str | None = None) -> dict[str, Any]:
    start_date = parse_date(start) if start else None
    end_date = parse_date(end) if end else None
    matched_ids: list[str] = []
    for record in records:
        record_date = parse_date(record.get("date"))
        if record_date is None:
            continue
        if start_date is not None and record_date < start_date:
            continue
        if end_date is not None and record_date > end_date:
            continue
        matched_ids.append(str(record.get("call_id")))
    return {"count": len(matched_ids), "start": start, "end": end, "record_ids": matched_ids[:20]}


def cohort_count(records: list[dict[str, Any]], date_granularity: str = "month") -> dict[str, int]:
    if date_granularity not in {"day", "month", "year"}:
        raise ValueError("cohort_count date_granularity must be day, month, or year")
    lengths = {"day": 10, "month": 7, "year": 4}
    counter: Counter[str] = Counter()
    for record in records:
        record_date = parse_date(record.get("date"))
        if record_date is None:
            continue
        counter[record_date.isoformat()[: lengths[date_granularity]]] += 1
    return sorted_counter(counter)


def ratio(numerator: int | float, denominator: int | float, precision: int = 4) -> dict[str, Any]:
    denominator_value = float(denominator)
    ratio_value = None if denominator_value == 0 else round(float(numerator) / denominator_value, int(precision))
    return {"numerator": numerator, "denominator": denominator, "ratio": ratio_value}


def field_values(record: dict[str, Any], field: str) -> list[Any]:
    value = record.get("fields", {}).get(field)
    if isinstance(value, list):
        return [item for item in value if item not in (None, "", False, "not_mentioned")]
    if value in (None, "", False, "not_mentioned"):
        return []
    return [value]


def value_matches(current: Any, expected: Any) -> bool:
    if isinstance(current, list):
        return expected in current
    return current == expected


def compare_value(current: Any, operator: str, expected: Any) -> bool:
    left = numeric_value(current)
    right = numeric_value(expected)
    if left is None or right is None:
        if operator == "==":
            return current == expected
        if operator == "!=":
            return current != expected
        return False
    if operator == "==":
        return left == right
    if operator == "!=":
        return left != right
    if operator == ">":
        return left > right
    if operator == ">=":
        return left >= right
    if operator == "<":
        return left < right
    if operator == "<=":
        return left <= right
    return False


def numeric_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def parse_date(value: Any) -> date | None:
    if not isinstance(value, str) or len(value) < 10:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def sort_nested_counts(value: dict[str, Any]) -> dict[str, Any]:
    sorted_items = sorted(
        value.items(),
        key=lambda item: (0, -item[1], item[0]) if isinstance(item[1], int) else (1, 0, item[0]),
    )
    return {
        key: sort_nested_counts(child) if isinstance(child, dict) else child
        for key, child in sorted_items
    }


def normalize_aggregation_result(result: Any) -> dict[str, Any]:
    if isinstance(result, Counter):
        return sorted_counter(result)
    if isinstance(result, dict):
        return {str(key): normalize_aggregation_value(value) for key, value in result.items()}
    if isinstance(result, int | float | str) or result is None:
        return {"result": result}
    if isinstance(result, list | tuple):
        return {"result": [normalize_aggregation_value(value) for value in result]}
    raise ValueError(f"Aggregation expression returned unsupported result type: {type(result).__name__}")


def normalize_aggregation_value(value: Any) -> Any:
    if isinstance(value, Counter):
        return sorted_counter(value)
    if isinstance(value, dict):
        return {str(key): normalize_aggregation_value(child) for key, child in value.items()}
    if isinstance(value, list | tuple):
        return [normalize_aggregation_value(item) for item in value]
    if isinstance(value, int | float | str) or value is None or isinstance(value, bool):
        return value
    return str(value)


def collect_aggregation_evidence_refs(
    records: list[dict[str, Any]],
    fields: list[str],
    *,
    limit: int = 24,
) -> list[str]:
    refs: list[str] = []
    seen: set[str] = set()
    for record in records:
        record_refs = record.get("evidence_refs", {})
        for field in fields:
            for ref in record_refs.get(field, []):
                if ref not in seen:
                    refs.append(ref)
                    seen.add(ref)
                if len(refs) >= limit:
                    return refs
    return refs
