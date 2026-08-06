/* Analytics — PostHog.
 *
 * Loaded by both pages so there is one place to change the config. Two rules
 * shape everything below, and both come from what this app handles: users paste
 * a provider API key into it, and they upload their CV.
 *
 *   1. Autocapture is OFF. Autocapture sends the text of whatever was clicked,
 *      and on /app that text is project descriptions rendered against someone's
 *      profile. Every event here is fired explicitly instead, so the property
 *      list is a thing we wrote rather than a thing we hope is safe.
 *   2. Session replay is OFF. It records the DOM, and the DOM holds CV text.
 *
 * What that leaves is counts and shapes — how many people arrive, how many
 * build a profile, how many finish a match, how long it took, which provider.
 * That is the funnel; none of it is anyone's content.
 */
(function () {
  var KEY = "phc_mPmFmCPUkdSGjenmDNRPTuFPMorbwc5Dj5wbEfdEZMN2";
  var HOST = "https://us.i.posthog.com";

  /* Official loader stub: queues calls until array.js lands, so `track()` is
     safe to call on the very first line of a page. */
  !function(t,e){var o,n,p,r;e.__SV||(window.posthog=e,e._i=[],e.init=function(i,s,a){function g(t,e){var o=e.split(".");2==o.length&&(t=t[o[0]],e=o[1]),t[e]=function(){t.push([e].concat(Array.prototype.slice.call(arguments,0)))}}(p=t.createElement("script")).type="text/javascript",p.crossOrigin="anonymous",p.async=!0,p.src=s.api_host.replace(".i.posthog.com","-assets.i.posthog.com")+"/static/array.js",(r=t.getElementsByTagName("script")[0]).parentNode.insertBefore(p,r);var u=e;for(void 0!==a?u=e[a]=[]:a="posthog",u.people=u.people||[],u.toString=function(t){var e="posthog";return"posthog"!==a&&(e+="."+a),t||(e+=" (stub)"),e},u.people.toString=function(){return u.toString(1)+".people (stub)"},o="init capture register register_once register_for_session unregister unregister_for_session getFeatureFlag getFeatureFlagPayload isFeatureEnabled reloadFeatureFlags updateEarlyAccessFeatureEnrollment getEarlyAccessFeatures on onFeatureFlags onSessionId getSurveys getActiveMatchingSurveys renderSurvey canRenderSurvey identify setPersonProperties group setPersonPropertiesForFlags resetPersonPropertiesForFlags setGroupPropertiesForFlags resetGroupPropertiesForFlags resetGroups reset get_distinct_id getGroups get_session_id get_session_replay_url alias set_config startSessionRecording stopSessionRecording sessionRecordingStarted captureException loadToolbar get_property getSessionProperty createPersonProfile opt_in_capturing opt_out_capturing has_opted_in_capturing has_opted_out_capturing clear_opt_in_out_capturing debug getPageViewId".split(" "),n=0;n<o.length;n++)g(u,o[n]);e._i.push([i,s,a])},e.__SV=1)}(document,window.posthog||[]);

  posthog.init(KEY, {
    api_host: HOST,
    defaults: "2025-05-24",

    // See rule 1 above. Pageviews and pageleaves still fire on their own.
    autocapture: false,
    capture_pageview: true,

    // See rule 2. Named explicitly rather than left to the project default, so
    // switching it on in the PostHog dashboard cannot quietly start recording
    // CVs — this line wins.
    disable_session_recording: true,

    // There is no login here, so every visitor would otherwise become a billed
    // "person" profile. Anonymous events still count uniques.
    person_profiles: "identified_only",

    // Belt and braces for rule 1: even if autocapture is ever turned back on,
    // element attributes (which include input values on some element types)
    // are not sent.
    mask_all_element_attributes: true,

    // Last line of defence. Runs on every event; drops any property whose value
    // looks like a provider key, whatever put it there.
    sanitize_properties: function (props) {
      for (var k in props) {
        if (typeof props[k] === "string" && /\b(sk-or-v1-|sk-ant-|sk-proj-|AIza)[A-Za-z0-9_\-]{8,}/.test(props[k])) {
          props[k] = "[redacted]";
        }
      }
      return props;
    },
  });

  /* Never let a blocked or failed analytics load break the app. uBlock and
     friends make posthog.capture throw, and a thrown tracker inside a click
     handler would take the button with it. */
  window.track = function (event, props) {
    try { window.posthog && posthog.capture(event, props || {}); } catch (_) {}
  };

  /* Provider errors are strings we do not control, so they are classified into
     a fixed set rather than sent through. Keeps the funnel readable and means a
     surprise upstream message cannot become event data. */
  window.errClass = function (message) {
    var m = String(message || "").toLowerCase();
    if (/429|rate|quota|exhaust/.test(m)) return "rate_limit";
    if (/401|403|api key|unauthor|denied/.test(m)) return "auth";
    if (/402|credit|billing|payment/.test(m)) return "credits";
    if (/404|not found|unknown model/.test(m)) return "bad_model";
    if (/parseable|json|empty response|no candidates/.test(m)) return "bad_output";
    if (/timeout|timed out|network|fetch/.test(m)) return "network";
    return "other";
  };
})();
