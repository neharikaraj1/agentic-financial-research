"""
Tests for financial calculator tool.
"""

import json
import pytest

from app.tools import financial_calculator_tool


def test_pe_ratio():
    data = json.dumps({"metric": "pe_ratio", "price": 150, "eps": 10})
    result = financial_calculator_tool.invoke({"data": data})
    assert "15.00" in result


def test_profit_margin():
    data = json.dumps({"metric": "profit_margin", "net_income": 20000, "revenue": 100000})
    result = financial_calculator_tool.invoke({"data": data})
    assert "20.00%" in result


def test_debt_to_equity():
    data = json.dumps({"metric": "debt_to_equity", "total_debt": 50000, "equity": 100000})
    result = financial_calculator_tool.invoke({"data": data})
    assert "0.50" in result


def test_current_ratio():
    data = json.dumps({"metric": "current_ratio", "current_assets": 200000, "current_liabilities": 100000})
    result = financial_calculator_tool.invoke({"data": data})
    assert "2.00" in result


def test_roe():
    data = json.dumps({"metric": "roe", "net_income": 15000, "equity": 100000})
    result = financial_calculator_tool.invoke({"data": data})
    assert "15.00%" in result


def test_cagr():
    data = json.dumps({"metric": "cagr", "start_value": 1000, "end_value": 2000, "years": 5})
    result = financial_calculator_tool.invoke({"data": data})
    assert "CAGR" in result


def test_eps():
    data = json.dumps({"metric": "eps", "net_income": 5000000, "shares": 1000000})
    result = financial_calculator_tool.invoke({"data": data})
    assert "5.00" in result


def test_division_by_zero():
    data = json.dumps({"metric": "pe_ratio", "price": 100, "eps": 0})
    result = financial_calculator_tool.invoke({"data": data})
    assert "zero" in result.lower() or "undefined" in result.lower()


def test_unknown_metric():
    data = json.dumps({"metric": "unknown_metric"})
    result = financial_calculator_tool.invoke({"data": data})
    assert "Unknown metric" in result


def test_invalid_json():
    result = financial_calculator_tool.invoke({"data": "not-valid-json"})
    assert "error" in result.lower()
