# Copyright Modal Labs 2026
import class_with_self_references
import class_with_self_references_impl
from typing_extensions import assert_type

wrapped_instance = class_with_self_references.SomeClass()
impl_instance = class_with_self_references_impl.SomeClass()

accept_self_result = wrapped_instance.accept_self(wrapped_instance)
assert_type(accept_self_result, class_with_self_references.SomeClass)

# xfail: Argument of type "SomeClass" cannot be assigned to parameter\
#        "s" of type "SomeClass" in function "accept_self"
wrapped_instance.accept_self(impl_instance)

accept_self_by_name_result = wrapped_instance.accept_self_by_name(wrapped_instance)
assert_type(accept_self_by_name_result, class_with_self_references.SomeClass)

sub_wrapped = class_with_self_references.SomeSubclass()
sub_self_result = sub_wrapped.accept_self(sub_wrapped)
assert_type(sub_self_result, class_with_self_references.SomeSubclass)

created_wrapped = class_with_self_references.SomeClass.create()
assert_type(created_wrapped, class_with_self_references.SomeClass)

# xfail: Argument of type "SomeClass" cannot be assigned to parameter\
#        "s" of type "SomeClass" in function "accept_self_by_name"
wrapped_instance.accept_self_by_name(impl_instance)  # this should fail
