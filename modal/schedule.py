from .object import Object
from .proto import api_pb2


class Schedule(Object, type_prefix="sc"):
    """A schedule of specific times at which registered functions are triggered"""

    @classmethod
    async def create(cls, period="", cron_string="", session=None):
        session = cls._get_session(session)

        if period:
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
            except:
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
