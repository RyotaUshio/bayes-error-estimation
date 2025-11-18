from __future__ import annotations

import dataclasses
import json
import pickle
import re
from pathlib import Path
from typing import Any, Literal, TypedDict

import numpy as np
import openreview
from pydantic import BaseModel, field_validator


def create_client_v1() -> openreview.Client:
    client = openreview.Client(
        baseurl='https://api.openreview.net',
    )
    return client


def create_client_v2() -> openreview.Client:
    client = openreview.api.OpenReviewClient(  # pyright: ignore
        baseurl='https://api2.openreview.net',
    )
    return client


def get_venue_id(year: int) -> str:
    if year > 2017:
        return f'ICLR.cc/{year}/Conference'
    if year == 2017:
        return 'ICLR.cc/2017/conference'
    raise ValueError(f'Unsupported year: {year}')


type Status = Literal[
    'desk-rejected', 'under-review', 'withdrawn', 'accepted', 'rejected'
]


class ICLRDataFetcher:
    client: openreview.Client
    year: int
    venue_id: str
    venue_group: openreview.Group
    is_v1: bool

    def __init__(self, year: int):
        self.client = create_client_v2()
        self.year = year
        self.venue_id = get_venue_id(year)
        self.venue_group = self.client.get_group(self.venue_id)
        self.is_v1 = self.venue_group.domain is None
        if self.is_v1:
            self.client = create_client_v1()

    def get_status_id(self, status: Status) -> str:
        # https://docs.openreview.net/how-to-guides/data-retrieval-and-modification/how-to-get-all-notes-for-submissions-reviews-rebuttals-etc#quickstart-getting-all-submissions
        status_key_map: dict[Status, str] = {
            'desk-rejected': 'desk_rejected_venue_id',
            'under-review': 'submission_venue_id ',
            'withdrawn': 'withdrawn_venue_id',
            'accepted': self.venue_id,
            'rejected': 'UNKNOWN',
        }
        status_key = status_key_map[status]
        status_id = self.venue_group.content[status_key]['value']  # type: ignore
        return status_id

    def get_submissions(self) -> map[Submission]:
        submissions = self.get_submissions_from_cache()
        if not submissions:
            submissions = self.fetch_submissions()
            self.cache_submissions(submissions)
        return map(
            lambda data: Submission(data, self.year, self.is_v1), submissions
        )

    def get_submission_cache_path(self) -> Path:
        return Path(f'data/iclr/cache/submissions_{self.year}.pkl')

    def get_submissions_from_cache(self) -> list[openreview.Note] | None:
        cache_path = self.get_submission_cache_path()
        try:
            with open(cache_path, 'rb') as f:
                submissions = pickle.load(f)
            return submissions
        except FileNotFoundError:
            return None

    def cache_submissions(self, submissions: list[openreview.Note]) -> None:
        cache_path = self.get_submission_cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(submissions, f)

    def fetch_submissions(self, **kwargs) -> list[openreview.Note]:
        if self.is_v1:
            # 2017 uses "submission" instead of "Blind_Submission"
            # (which is undocumented!)
            submission_name = (
                'submission' if self.year == 2017 else 'Blind_Submission'
            )
            details = 'directReplies'
        else:
            submission_name = self.venue_group.content['submission_name'][  # type: ignore
                'value'
            ]
            details = 'replies'

        submissions = self.client.get_all_notes(
            invitation=f'{self.venue_id}/-/{submission_name}',
            details=details,
            **kwargs,
        )

        assert type(submissions) is list
        assert len(submissions) > 0

        return submissions

    def get_types(self) -> set[str]:
        types = set()

        for submission in self.get_submissions():
            for reply in submission.replies:
                if self.is_v1:
                    if self.year == 2017:
                        parts = reply['invitation'].split('/')[-2:]
                        if parts[0].startswith('paper'):
                            types.add(parts[1])
                        else:
                            types.add('/'.join(parts))
                    else:
                        types.add(reply['invitation'].split('/')[-1])
                else:
                    for invitation in reply['invitations']:
                        types.add(invitation.split('/')[-1])

        return types


class Submission:
    data: openreview.Note
    year: int
    is_v1: bool
    _reviews: Reviews | None = None
    _is_accepted: bool | None = None

    def __init__(self, data: openreview.Note, year: int, is_v1: bool):
        self.data = data
        self.year = year
        self.is_v1 = is_v1

    @property
    def replies(self) -> list[Any]:
        return self.data.details[  # type: ignore
            'directReplies' if self.is_v1 else 'replies'
        ]

    @property
    def reviews(self) -> Reviews:
        if self._reviews is None:
            self._reviews = Reviews.of(self)
        return self._reviews

    @property
    def is_accepted(self) -> bool:
        if self._is_accepted is not None:
            return self._is_accepted

        if self.is_v1:
            # can be "Accept (*)" or "Reject"
            # For 2017 and 2018, it also can be "Invite to Workshop Track"
            decision = self.get_decision() or self.get_metareview()
            self._is_accepted = bool(
                decision and decision.lower().startswith('accept')
            )
            return self._is_accepted

        # see https://docs.openreview.net/how-to-guides/data-retrieval-and-modification/how-to-get-all-notes-for-submissions-reviews-rebuttals-etc#quickstart-getting-all-submissions
        venue_id: str | None = get_value(self.data.content['venueid'])
        self._is_accepted = venue_id == get_venue_id(self.year)
        return self._is_accepted

    def is_reply_type(self, reply: Any, type: str) -> bool:
        if self.is_v1:
            return reply['invitation'].endswith('/' + type)
        else:
            return any(
                invitation.endswith('/' + type)
                for invitation in reply['invitations']
            )

    def is_decision(self, reply: Any) -> bool:
        match self.year:
            case 2017:
                name = 'acceptance'
            case 2018:
                name = 'Acceptance_Decision'
            case _:
                name = 'Decision'

        return self.is_reply_type(reply, name)

    def get_decision(self) -> str | None:
        for reply in self.replies:
            if self.is_decision(reply):
                decision = reply['content']['decision']
                return get_value(decision)
        return None

    def is_metareview(self, reply: Any) -> bool:
        return self.is_reply_type(reply, 'Meta_Review')

    def get_metareview(self) -> str | None:
        for reply in self.replies:
            if self.is_metareview(reply):
                metareview = reply['content']['recommendation']
                return get_value(metareview)
        return None

    def to_label_pair(self) -> LabelPair | None:
        soft_label = self.reviews.normalized_rating
        if soft_label is None:
            return None
        hard_label = 1 if self.is_accepted else 0
        return LabelPair(hard_label=hard_label, soft_label=soft_label)


class LabelPair(TypedDict):
    hard_label: Literal[0, 1]
    soft_label: float


def get_rating_range(year: int) -> tuple[float, float]:
    return (1.0, 8.0) if year == 2020 else (1.0, 10.0)


def get_confidence_range(year: int) -> tuple[float, float] | None:
    return None if year == 2020 else (1.0, 5.0)


def get_default_confidence(year: int) -> float:
    range = get_confidence_range(year)
    if range is None:
        return 3.0
    min, max = range
    return (min + max) / 2


@dataclasses.dataclass
class Reviews:
    reviews: list[Review]
    year: int

    @staticmethod
    def of(submission: Submission) -> Reviews:
        reviews = []
        for reply in submission.replies:
            if Review.is_review(submission, reply):
                review = Review.from_reply(reply)
                if review:
                    reviews.append(review)
        return Reviews(reviews, submission.year)

    @property
    def normalized_rating(self) -> float | None:
        avg_rating = self.average_rating
        if avg_rating is None:
            return None

        min_rating, max_rating = get_rating_range(self.year)
        return (avg_rating - min_rating) / (max_rating - min_rating)

    @property
    def average_rating(self) -> float | None:
        if not self.reviews:
            return None
        default_confidence = get_default_confidence(self.year)
        confidences = [
            review.confidence
            if review.confidence is not None
            else default_confidence
            for review in self.reviews
        ]
        total_confidence = sum(confidences)
        weighted_ratings = sum(
            review.rating * confidence
            for review, confidence in zip(self.reviews, confidences)
        )
        return weighted_ratings / total_confidence

    def __iter__(self):
        return iter(self.reviews)

    def __len__(self):
        return len(self.reviews)


pattern = re.compile(r'(\d+): ')


@dataclasses.dataclass
class Review:
    rating: float
    confidence: float | None

    @staticmethod
    def is_review(submission: Submission, reply: Any) -> bool:
        return submission.is_reply_type(
            reply,
            'official/review' if submission.year == 2017 else 'Official_Review',
        )

    @staticmethod
    def from_reply(reply: Any) -> Review | None:
        rating = Review.get_rating(reply)
        confidence = Review.get_confidence(reply)

        if not rating:
            return None

        return Review(rating=rating, confidence=confidence)

    @staticmethod
    def get_rating(reply: Any) -> float | None:
        content = reply['content']
        rating = get_value(content.get('rating', {})) or content.get(
            'recommendation'
        )
        if isinstance(rating, (int, float)):
            return float(rating)

        if not isinstance(rating, str):
            return None

        rating_match = pattern.match(rating)
        if rating_match is None:
            raise ValueError(f'Invalid rating format: {rating}')

        return float(rating_match.group(1))

    @staticmethod
    def get_confidence(reply: Any) -> float | None:
        content = reply['content']
        confidence = get_value(content.get('confidence'))

        if isinstance(confidence, (int, float)):
            return float(confidence)

        if not isinstance(confidence, str):
            return None

        confidence_match = pattern.match(confidence)
        if confidence_match is None:
            raise ValueError(f'Invalid confidence format: {confidence}')

        return float(confidence_match.group(1))


def get_value[T](x: dict[Literal['value'], T] | T) -> T | None:
    return x.get('value') if isinstance(x, dict) else x


def get_iclr_data_path(year: int) -> Path:
    return Path(f'data/iclr/{year}.json')


def fetch_iclr_year(year: int):
    iclr = ICLRDataFetcher(year)
    label_pairs: list[LabelPair] = []

    for submission in iclr.get_submissions():
        label_pair = submission.to_label_pair()

        # filter out submissions with no reviews
        # (e.g., withdrawn before review, desk-rejected)
        if label_pair is None:
            continue

        label_pairs.append(label_pair)

    with open(get_iclr_data_path(year), 'w') as f:
        json.dump(label_pairs, f, indent=2)

    return label_pairs


def load_iclr_year(year: int):
    path = get_iclr_data_path(year)
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return data
    except Exception:
        data = fetch_iclr_year(year)
        return data


class ICLROptions(BaseModel):
    dataset: Literal['iclr']
    years: list[int] = list(range(2017, 2026))

    @field_validator('years', mode='before')
    @classmethod
    def validate_years(cls, val):
        if isinstance(val, str):
            years = set()
            for part in val.split(','):
                if '-' in part:
                    start, end = part.split('-')
                    years.update(
                        range(int(start or 2017), int(end or 2025) + 1)
                    )
                elif part:
                    years.add(int(part))
            return list(years)


def load_iclr(options: ICLROptions):
    label_pairs = []

    for year in options.years:
        label_pairs += load_iclr_year(year)

    soft_labels_corrupted = []
    labels = []
    for pair in label_pairs:
        soft_labels_corrupted.append(pair['soft_label'])
        labels.append(pair['hard_label'])

    return {
        'corrupted': {
            'soft_labels': np.array(soft_labels_corrupted),
            'labels': np.array(labels),
        }
    }
