package modal

import (
	"fmt"
	"regexp"
	"strings"
)

var (
	objectNameCharset      = regexp.MustCompile(`^[a-zA-Z0-9\-_.]+$`)
	appIDPattern           = regexp.MustCompile(`^ap-[a-zA-Z0-9]{22}$`)
	environmentNameCharset = regexp.MustCompile(`^[a-zA-Z0-9][a-zA-Z0-9_.-]*$`)
)

// checkEnvironmentName allows periods when looking up existing Environments.
// Newly created Environment names cannot contain periods.
func checkEnvironmentName(name string, allowPeriods bool) error {
	if len(name) <= 64 && environmentNameCharset.MatchString(name) &&
		!strings.HasPrefix(strings.ToLower(name), "en-") && (allowPeriods || !strings.Contains(name, ".")) {
		return nil
	}
	allowed := "alphanumeric characters, dashes, and underscores"
	if allowPeriods {
		allowed = "alphanumeric characters, dashes, periods, and underscores"
	}
	return InvalidError{fmt.Sprintf("Invalid Environment name: '%s'. Environment names must be 1-64 characters, start with an alphanumeric character, contain only %s, and cannot start with 'en-'.", name, allowed)}
}

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
