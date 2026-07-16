import asyncio

from search_planner import SearchQueryPlanner, merge_search_results, normalize_result_url


class FakeLLMClient:
    def __init__(self, response=None, exc=None):
        self.response = response
        self.exc = exc
        self.calls = []

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        if self.exc:
            raise self.exc
        return self.response


def test_planner_parses_json_and_uses_deterministic_settings():
    async def run():
        llm = FakeLLMClient(
            '{"queries":["RTX 5090 current price site:ebay.com","RTX 5090 sold listings site:ebay.com"],'
            '"reason":"Current eBay pricing is relevant."}'
        )
        planner = SearchQueryPlanner(llm)

        plan = await planner.plan("Kolik teď stojí 5090 na ebay? Hoď mi nějaký link")

        assert plan.queries == [
            "RTX 5090 current price site:ebay.com",
            "RTX 5090 sold listings site:ebay.com",
        ]
        assert plan.fallback_used is False
        assert llm.calls[0]["temperature"] == 0.0
        assert llm.calls[0]["max_tokens"] == 200
        assert llm.calls[0]["stream"] is False
        assert llm.calls[0]["image_url"] is None

    asyncio.run(run())


def test_parse_plan_validates_deduplicates_and_limits_queries():
    planner = SearchQueryPlanner(FakeLLMClient("{}"))
    plan = planner.parse_plan(
        '{"queries":["one","one","two","three","four"],"reason":"test"}'
    )

    assert plan.queries == ["one", "two", "three"]
    assert plan.reason == "test"


def test_fallback_for_malformed_json_preserves_website_specific_request():
    async def run():
        planner = SearchQueryPlanner(FakeLLMClient("not json"))

        plan = await planner.plan("Kolik teď stojí 5090 na ebay? Hoď mi nějaký link")

        assert plan.fallback_used is True
        assert "RTX 5090 current price site:ebay.com" in plan.queries
        assert "RTX 5090 sold listings site:ebay.com" in plan.queries

    asyncio.run(run())


def test_fallback_for_specific_person_uses_quoted_name():
    async def run():
        planner = SearchQueryPlanner(FakeLLMClient(exc=RuntimeError("backend down")))

        plan = await planner.plan("Čím se proslavil politik Miroslav Sládek")

        assert plan.fallback_used is True
        assert '"Miroslav Sládek" political career' in plan.queries
        assert '"Miroslav Sládek" biography' in plan.queries

    asyncio.run(run())


def test_clean_query_removes_filler_and_mentions():
    planner = SearchQueryPlanner(FakeLLMClient("{}"))

    cleaned = planner.clean_query("<@123> please give me a link what is the latest stable vllm version")

    assert "<@123>" not in cleaned
    assert "please" not in cleaned.lower()
    assert "give me" not in cleaned.lower()
    assert "vLLM" in cleaned


def test_merge_search_results_deduplicates_by_normalized_url():
    groups = [
        [
            {"title": "A", "url": "https://www.example.com/page/", "snippet": "first"},
            {"title": "B", "url": "https://other.example/item", "snippet": "second"},
        ],
        [
            {"title": "A duplicate", "url": "https://example.com/page", "snippet": "duplicate"},
            {"title": "C", "url": "https://third.example/item", "snippet": "third"},
        ],
    ]

    merged = merge_search_results(groups, limit=10)

    assert [result["title"] for result in merged] == ["A", "B", "C"]


def test_normalize_result_url_removes_www_and_trailing_slash():
    assert normalize_result_url("https://www.example.com/path/") == "https://example.com/path"