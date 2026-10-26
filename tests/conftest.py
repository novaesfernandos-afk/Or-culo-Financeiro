from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest

import oraculo.skills  # noqa: F401 - registra as skills
from oraculo.core.types import Frequency, NewsItem, Series, utcnow
from oraculo.data.providers.synthetic import business_days
from oraculo.skills.base import SkillContext

TEST_SOURCE = "Fixture de teste"


def make_series(values, name="price:TEST", end: date | None = None, freq=Frequency.DAILY, synthetic=False, source=TEST_SOURCE) -> Series:
    values = np.asarray(values, float)
    dates = business_days(end or date.today(), len(values))
    return Series(name=name, dates=dates, values=values, source=source, retrieved_at=utcnow(), frequency=freq, synthetic=synthetic)


def gbm(n=800, mu=0.0003, sigma=0.015, seed=1, start=20.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return start * np.exp(np.cumsum(rng.normal(mu, sigma, n)))


def news(title, source="Reuters", url="https://www.reuters.com/x", body="", hours_ago=1, tickers=None) -> NewsItem:
    now = datetime.now(timezone.utc)
    return NewsItem(title=title, source=source, published_at=now - timedelta(hours=hours_ago), retrieved_at=now, body=body, url=url, tickers=tickers or [])


@pytest.fixture
def ctx() -> SkillContext:
    return SkillContext()


@pytest.fixture
def prices() -> Series:
    return make_series(gbm())


@pytest.fixture
def svc():
    from oraculo.modes import Services

    return Services.create(demo=True)
