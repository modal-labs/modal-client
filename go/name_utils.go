package modal

import (
	"fmt"
	"regexp"
)

var (
	objectNameCharset = regexp.MustCompile(`^[a-zA-Z0-9\-_.]+$`)
	appIDPattern      = regexp.MustCompile(`^ap-[a-zA-Z0-9]{22}$`)
)

// isValidObjectName checks whether a name is a valid Modal object name.
func isValidObjectName(name string) bool {
	return len(name) <= 64 &&
		objectNameCharset.MatchString(name) &&
		!appIDPattern.MatchString(name)
}

// checkObjectName validates a Modal object name, returning an InvalidError
// if the name is not valid. The objectType is used in the error message
// (e.g. "Image", "Image tag").
func checkObjectName(name string, objectType string) error {
	if !isValidObjectName(name) {
		return InvalidError{
			Exception: fmt.Sprintf(
				"Invalid %s name: '%s'."+
					"\n\nNames may contain only alphanumeric characters, dashes, periods, and underscores,"+
					" must be shorter than 64 characters, and cannot conflict with App ID strings.",
				objectType, name,
			),
		}
	}
	return nil
}
