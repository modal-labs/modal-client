from .object import Object
from .proto import api_pb2


class Schedule(Object, type_prefix="sc"):
    """A schedule of specific times at which registered functions are triggered"""

    @classmethod
    async def create(cls, period=None, cron_string=None, session=None):
        session = cls._get_session(session)

        if period:
            # TODO: should we just take a timedelta object?
            seconds = None
            try:
                if period[-1] == "d":
                    seconds = int(period[:-1]) * 24 * 3600
                elif period[-1] == "h":
                    seconds = int(period[:-1]) * 3600
                elif period[-1] == "m":
                    seconds = int(period[:-1]) * 60
                elif period[-1] == "s":
                    seconds = int(period[:-1])
                else:
                    raise
            except Exception:
                raise Exception(f"Failed to parse period while creating Schedule: {period}")
            req = api_pb2.ScheduleCreateRequest(
                session_id=session.session_id,
                period=seconds,
                args=session.serialize([]),
                kwargs=session.serialize({}),
            )
        elif cron_string:
            req = api_pb2.ScheduleCreateRequest(
                session_id=session.session_id,
                cron_string=cron_string,
                args=session.serialize([]),
                kwargs=session.serialize({}),
            )

        # TODO: remove args/kwargs placeholders, which represent arguments to scheduled function
        # This would involve requesting an output buffer here similar to _Invocation

        resp = await session.client.stub.ScheduleCreate(req)

        if resp.error_message:
            raise Exception(f"Failed to create Schedule: {resp.error_message}")

        schedule_id = resp.schedule_id
        return cls._create_object_instance(schedule_id, session)

    @classmethod
    def period(cls, period):
        return _period(period)

    @classmethod
    def cron(cls, cron_string):
        return _cron(cron_string)


# TODO(erikbern): methods below seem like a bit unnecessary layer of indirection
# We need them because we can't create factories on the class itself since the
# class doesn't exist at that point. This seems like a solvable problem.


@Schedule.factory
async def _period(period):
    return await Schedule.create(period=period)


@Schedule.factory
async def _cron(cron_string):
    return await Schedule.create(cron_string=cron_string)
