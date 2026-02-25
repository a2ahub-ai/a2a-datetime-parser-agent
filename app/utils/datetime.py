from datetime import datetime, timedelta
from typing import Optional
from pydantic import BaseModel, Field, model_validator


class AbsoluteTime(BaseModel):
    """Represents absolute time values."""
    year: Optional[int] = None
    month: Optional[int] = None
    day: Optional[int] = None
    hour: Optional[int] = None
    minute: Optional[int] = None


class RelativeTime(BaseModel):
    """Represents relative time offsets."""
    year: Optional[int] = None
    month: Optional[int] = None
    day: Optional[int] = None
    hour: Optional[int] = None
    minute: Optional[int] = None


class TimeSingle(BaseModel):
    """Represents a single point in time."""
    absolute: Optional[AbsoluteTime] = None
    relative: Optional[RelativeTime] = None
    now: Optional[bool] = None


class TimeRangeDate(BaseModel):
    """Represents a date in a time range."""
    absolute: Optional[AbsoluteTime] = None
    relative: Optional[RelativeTime] = None
    now: Optional[bool] = None


class TimeRange(BaseModel):
    """Represents a time range with start and end dates."""
    start_date: TimeRangeDate = Field(alias="start_date")
    end_date: TimeRangeDate = Field(alias="end_date")

    class Config:
        populate_by_name = True


class ComputedDateTime(BaseModel):
    """Represents a computed date/time result."""
    now: Optional[bool] = None
    datetime: Optional[str] = None

    class Config:
        # Exclude None values from the serialized output
        exclude_none = True


class TimeInputPayload(BaseModel):
    """Input payload for time conversion.

    Note: Only one of time_single or time_range can be set at a time.
    """
    time_single: Optional[TimeSingle] = None
    time_range: Optional[TimeRange] = None

    @model_validator(mode='after')
    def validate_exclusive_fields(self):
        """Ensure only one of time_single or time_range is set."""
        if self.time_single is not None and self.time_range is not None:
            raise ValueError(
                "Only one of 'time_single' or 'time_range' can be set at a time, not both."
            )
        return self


class TimeConvertedPayload(BaseModel):
    """Output payload after time conversion.

    Note: Only one of time_single or time_range will be set.
    """
    parsable: bool
    reason: Optional[str] = None
    time_single: Optional[ComputedDateTime] = None
    time_range: Optional[dict[str, ComputedDateTime]] = None


def convert_datetime_payload(payload: TimeInputPayload, current_date_str: str) -> TimeConvertedPayload:
    """
    Convert datetime payload from Google Home format to Tuya format.

    Args:
        payload: The input time payload
        current_date_str: The current datetime as a string (ISO format)

    Returns:
        TimeConvertedPayload with converted time information
    """
    current = datetime.fromisoformat(current_date_str.replace('Z', '+00:00'))

    def is_empty_time_object(time_obj: Optional[TimeSingle | TimeRangeDate]) -> bool:
        """Check if a time object is None or empty."""
        if time_obj is None:
            return True
        if time_obj.now:
            return False
        if time_obj.absolute:
            abs_time = time_obj.absolute
            if any([abs_time.year is not None, abs_time.month is not None, abs_time.day is not None,
                    abs_time.hour is not None, abs_time.minute is not None]):
                return False
        if time_obj.relative:
            rel_time = time_obj.relative
            if any([rel_time.year is not None, rel_time.month is not None, rel_time.day is not None,
                    rel_time.hour is not None, rel_time.minute is not None]):
                return False
        return True

    def has_time_units(t: Optional[TimeSingle | TimeRangeDate]) -> bool:
        """Check if a time object has explicit time units (hour/minute)."""
        if not t:
            return False
        abs_time = t.absolute or AbsoluteTime()
        rel_time = t.relative or RelativeTime()
        return (abs_time.hour is not None or abs_time.minute is not None or
                rel_time.hour is not None or rel_time.minute is not None)

    def build_expanded_endpoint(t_obj: Optional[TimeRangeDate], is_start: bool) -> ComputedDateTime:
        """Build an expanded endpoint for a time range when no explicit time units."""
        # Check for 'now' flag first
        if t_obj and t_obj.now:
            return ComputedDateTime(now=True)

        base_start = datetime(current.year, current.month, current.day,
                              current.hour, current.minute, current.second)
        base_end = datetime(current.year, current.month, current.day,
                            current.hour, current.minute, current.second)

        if not t_obj:
            # No object provided -> return full-day for current day
            base_start = base_start.replace(hour=0, minute=0, second=0, microsecond=0)
            base_end = base_end.replace(hour=23, minute=59, second=59, microsecond=999000)
            chosen = base_start if is_start else base_end
            return ComputedDateTime(
                datetime=chosen.strftime('%Y-%m-%dT%H:%M:%S')
            )

        # Apply relative shifts first
        if t_obj.relative:
            rel = t_obj.relative
            if rel.year is not None:
                base_start = base_start.replace(year=base_start.year + rel.year)
                base_end = base_end.replace(year=base_end.year + rel.year)
            if rel.month is not None:
                # Handle month overflow
                new_month_start = base_start.month + rel.month
                new_month_end = base_end.month + rel.month
                year_offset_start = (new_month_start - 1) // 12
                year_offset_end = (new_month_end - 1) // 12
                base_start = base_start.replace(
                    year=base_start.year + year_offset_start,
                    month=((new_month_start - 1) % 12) + 1
                )
                base_end = base_end.replace(
                    year=base_end.year + year_offset_end,
                    month=((new_month_end - 1) % 12) + 1
                )
            if rel.day is not None:
                base_start += timedelta(days=rel.day)
                base_end += timedelta(days=rel.day)
            if rel.hour is not None:
                base_start += timedelta(hours=rel.hour)
                base_end += timedelta(hours=rel.hour)
            if rel.minute is not None:
                base_start += timedelta(minutes=rel.minute)
                base_end += timedelta(minutes=rel.minute)

        # Then apply absolute overrides (if present)
        if t_obj.absolute:
            abs_time = t_obj.absolute
            if abs_time.year is not None:
                base_start = base_start.replace(year=abs_time.year)
                base_end = base_end.replace(year=abs_time.year)
            if abs_time.month is not None:
                base_start = base_start.replace(month=abs_time.month)
                base_end = base_end.replace(month=abs_time.month)
            if abs_time.day is not None:
                base_start = base_start.replace(day=abs_time.day)
                base_end = base_end.replace(day=abs_time.day)
            if abs_time.hour is not None or abs_time.minute is not None:
                h = abs_time.hour if abs_time.hour is not None else base_start.hour
                m = abs_time.minute if abs_time.minute is not None else (
                    0 if abs_time.hour is not None else base_start.minute)
                base_start = base_start.replace(hour=h, minute=m, second=0, microsecond=0)
                base_end = base_end.replace(hour=h, minute=m, second=0, microsecond=0)

        # Determine if the input mentioned day/month/year
        abs_time = t_obj.absolute or AbsoluteTime()
        rel_time = t_obj.relative or RelativeTime()
        has_day = abs_time.day is not None or rel_time.day is not None
        has_month = abs_time.month is not None or rel_time.month is not None
        has_year = abs_time.year is not None or rel_time.year is not None

        # Special case: if only relative.day=0 (no other fields), treat as current moment
        if (has_day and not has_month and not has_year and
            rel_time.day == 0 and abs_time.day is None and
            rel_time.month is None and rel_time.year is None and
                abs_time.month is None and abs_time.year is None):
            # Return current moment, not full day range
            chosen = base_start if is_start else base_end
            return ComputedDateTime(datetime=chosen.strftime('%Y-%m-%dT%H:%M:%S'))

        # Build range when no explicit time units
        if has_day:
            base_start = base_start.replace(hour=0, minute=0, second=0, microsecond=0)
            base_end = base_end.replace(hour=23, minute=59, second=59, microsecond=999000)
        elif has_month:
            # Start = first day of month 00:00:00
            base_start = base_start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

            # End = last day of month 23:59:59
            if base_end.month == 12:
                last_day = datetime(base_end.year + 1, 1, 1) - timedelta(days=1)
            else:
                last_day = datetime(base_end.year, base_end.month + 1, 1) - timedelta(days=1)
            base_end = base_end.replace(day=last_day.day, hour=23, minute=59, second=59, microsecond=999000)
        elif has_year:
            # Start = Jan 1 00:00:00
            base_start = base_start.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)

            # End = Dec 31 23:59:59
            base_end = base_end.replace(month=12, day=31, hour=23, minute=59, second=59, microsecond=999000)
        else:
            # No day/month/year specified: treat as full current day
            base_start = base_start.replace(hour=0, minute=0, second=0, microsecond=0)
            base_end = base_end.replace(hour=23, minute=59, second=59, microsecond=999000)

        # Choose which endpoint to return
        chosen = base_start if is_start else base_end
        return ComputedDateTime(datetime=chosen.strftime('%Y-%m-%dT%H:%M:%S'))

    # Check if the payload is parsable
    parsable = True
    reason = None

    # Handle case when neither time_single nor time_range is provided
    if not payload.time_single and not payload.time_range:
        parsable = False
        reason = "No datetime information provided in the input"
        return TimeConvertedPayload(
            parsable=parsable,
            reason=reason
        )

    # Handle time_single
    if payload.time_single:
        if is_empty_time_object(payload.time_single):
            parsable = False
            reason = "time_single is empty or contains no datetime data"
            return TimeConvertedPayload(
                parsable=parsable,
                reason=reason
            )

        time_single_result = compute_date_time(payload.time_single, current)
        return TimeConvertedPayload(
            parsable=parsable,
            reason=reason,
            time_single=time_single_result
        )

    elif payload.time_range:
        # Handle time_range
        tr = payload.time_range

        # Check if both start and end are empty
        start_empty = is_empty_time_object(tr.start_date)
        end_empty = is_empty_time_object(tr.end_date)

        if start_empty and end_empty:
            parsable = False
            reason = "Both start_date and end_date are empty or contain no datetime data"

        # Process start and end endpoints
        start_has_time = has_time_units(tr.start_date)
        end_has_time = has_time_units(tr.end_date)

        if start_has_time:
            start_computed = compute_date_time(tr.start_date, current)
        else:
            start_computed = build_expanded_endpoint(tr.start_date, True)

        if end_has_time:
            end_computed = compute_date_time(tr.end_date, current)
        else:
            end_computed = build_expanded_endpoint(tr.end_date, False)

        return TimeConvertedPayload(
            parsable=parsable,
            reason=reason,
            time_range={
                'start_date': start_computed,
                'end_date': end_computed
            }
        )

    # Fallback case (should not be reached)
    return TimeConvertedPayload(
        parsable=False,
        reason="Invalid payload structure"
    )


def compute_date_time(time_obj: TimeSingle | TimeRangeDate, current: datetime) -> ComputedDateTime:
    """
    Compute a single date/time from a time object.

    Args:
        time_obj: The time specification object
        current: The current datetime

    Returns:
        ComputedDateTime with the computed date and optionally time
    """
    if time_obj.now:
        return ComputedDateTime(now=True)

    dt = datetime(current.year, current.month, current.day,
                  current.hour, current.minute, current.second)

    # Apply relative shifts
    if time_obj.relative:
        rel = time_obj.relative
        if rel.year is not None:
            dt = dt.replace(year=dt.year + rel.year)
        if rel.month is not None:
            new_month = dt.month + rel.month
            year_offset = (new_month - 1) // 12
            dt = dt.replace(
                year=dt.year + year_offset,
                month=((new_month - 1) % 12) + 1
            )
        if rel.day is not None:
            dt += timedelta(days=rel.day)
        if rel.hour is not None:
            dt += timedelta(hours=rel.hour)
        if rel.minute is not None:
            dt += timedelta(minutes=rel.minute)

    # Apply absolute overrides
    if time_obj.absolute:
        abs_time = time_obj.absolute
        if abs_time.year is not None:
            dt = dt.replace(year=abs_time.year)
        if abs_time.month is not None:
            dt = dt.replace(month=abs_time.month)
        if abs_time.day is not None:
            dt = dt.replace(day=abs_time.day)

        if abs_time.hour is not None or abs_time.minute is not None:
            h = abs_time.hour if abs_time.hour is not None else dt.hour
            m = abs_time.minute if abs_time.minute is not None else (0 if abs_time.hour is not None else dt.minute)
            dt = dt.replace(hour=h, minute=m, second=0, microsecond=0)

    # Check if time units were specified
    has_time = False
    if time_obj.absolute:
        has_time = (time_obj.absolute.hour is not None or time_obj.absolute.minute is not None)
    if time_obj.relative and not has_time:
        has_time = (time_obj.relative.hour is not None or time_obj.relative.minute is not None)

    # Always return datetime in ISO format
    return ComputedDateTime(datetime=dt.strftime('%Y-%m-%dT%H:%M:%S'))
