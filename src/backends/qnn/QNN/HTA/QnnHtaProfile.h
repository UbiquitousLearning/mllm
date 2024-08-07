//==============================================================================
//
//  Copyright (c) 2022 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

/**
 *  @file
 *  @brief QNN HTA Profile component API.
 *
 *          Requires HTA backend to be initialized.
 *          Should be used with the QnnProfile API but has HTA backend
 *          specific definition for different QnnProfile data structures
 *
 */

#ifndef QNN_HTA_PROFILE_H
#define QNN_HTA_PROFILE_H

#include "QnnProfile.h"

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Macros
//=============================================================================
/**
 * @brief QnnProfile_EventType_t definition to get profile information
 *        that corresponds to the remote procedure call on the ARM processor
 *        when client invokes QnnContext_createFromBinary. The value
 *        returned is time in microseconds.
 *
 * @note context load binary host time maybe available on both
 *       QNN_PROFILE_LEVEL_BASIC and QNN_PROFILE_LEVEL_DETAILED levels
 */
#define QNN_HTA_PROFILE_EVENTTYPE_CONTEXT_LOAD_BIN_HOST_TIME_MICROSEC 1002

/**
 * @brief QnnProfile_EventType_t definition to get profile information
 *        that corresponds to the remote procedure call on the HTA processor
 *        when client invokes QnnContext_createFromBinary. The value
 *        returned is time in microseconds.
 *
 * @note context load binary HTA time maybe available on both
 *       QNN_PROFILE_LEVEL_BASIC and QNN_PROFILE_LEVEL_DETAILED levels
 */
#define QNN_HTA_PROFILE_EVENTTYPE_CONTEXT_LOAD_BIN_HTA_TIME_MICROSEC 1003

/**
 * @brief QnnProfile_EventType_t definition to get profile information
 *        that corresponds to the time taken to create the context on the
 *        accelerator when client invokes QnnContext_createFromBinary.
 *        The value returned is time in microseconds.
 *
 * @note context load binary accelerator time maybe available on both
 *       QNN_PROFILE_LEVEL_BASIC and QNN_PROFILE_LEVEL_DETAILED levels
 */
#define QNN_HTA_PROFILE_EVENTTYPE_CONTEXT_LOAD_BIN_ACCEL_TIME_MICROSEC 1004

/**
 * @brief QnnProfile_EventType_t definition to get profile information
 *        that corresponds to the remote procedure call on the ARM processor
 *        when client invokes QnnGraph_finalize.
 *        The value returned is time in microseconds.
 *
 * @note graph finalize host time maybe available on both
 *       QNN_PROFILE_LEVEL_BASIC and QNN_PROFILE_LEVEL_DETAILED levels
 */
#define QNN_HTA_PROFILE_EVENTTYPE_GRAPH_FINALIZE_HOST_TIME_MICROSEC 2001

/**
 * @brief QnnProfile_EventType_t definition to get profile information
 *        that corresponds to the remote procedure call on the HTA processor
 *        when client invokes QnnGraph_finalize.
 *        The value returned is time in microseconds.
 *
 * @note graph finalize HTA time maybe available on both
 *       QNN_PROFILE_LEVEL_BASIC and QNN_PROFILE_LEVEL_DETAILED levels
 */
#define QNN_HTA_PROFILE_EVENTTYPE_GRAPH_FINALIZE_HTA_TIME_MICROSEC 2002

/**
 * @brief QnnProfile_EventType_t definition to get profile information
 *        that corresponds to finalize the graph on the accelerator
 *        when client invokes QnnGraph_finalize.
 *        The value returned is time in microseconds.
 *
 * @note graph finalize accelerator time maybe available on both
 *       QNN_PROFILE_LEVEL_BASIC and QNN_PROFILE_LEVEL_DETAILED levels
 */
#define QNN_HTA_PROFILE_EVENTTYPE_GRAPH_FINALIZE_ACCEL_TIME_MICROSEC 2003

/**
 * @brief QnnProfile_EventType_t definition to get profile information
 *        that corresponds to the remote procedure call on the ARM processor
 *        when client invokes QnnGraph_execute or QnnGraph_executeAsync.
 *        The value returned is time in microseconds.
 *
 * @note graph execute host time maybe available on both
 *       QNN_PROFILE_LEVEL_BASIC and QNN_PROFILE_LEVEL_DETAILED levels
 */
#define QNN_HTA_PROFILE_EVENTTYPE_GRAPH_EXECUTE_HOST_TIME_MICROSEC 3001

/**
 * @brief QnnProfile_EventType_t definition to get profile information
 *        that corresponds to the remote procedure call on the HTA processor
 *        when client invokes QnnGraph_execute or QnnGraph_executeAsync.
 *        The value returned is time in microseconds.
 *
 * @note graph execute HTA time maybe available on both
 *       QNN_PROFILE_LEVEL_BASIC and QNN_PROFILE_LEVEL_DETAILED levels
 */
#define QNN_HTA_PROFILE_EVENTTYPE_GRAPH_EXECUTE_HTA_TIME_MICROSEC 3002

/**
 * @brief QnnProfile_EventType_t definition to get profile information
 *        that corresponds to execute the graph on the accelerator
 *        when client invokes QnnGraph_execute or QnnGraph_executeAsync.
 *        The value returned is number of processor cycles taken.
 *
 * @note graph execute accelerator time maybe available only on
 *       QNN_PROFILE_LEVEL_DETAILED levels
 *
 * @note When QNN_PROFILE_LEVEL_DETAILED is used, this event can have
 *       multiple sub-events of type QNN_PROFILE_EVENTTYPE_NODE.
 *       There will be a sub-event for each node that was added to the graph
 */
#define QNN_HTA_PROFILE_EVENTTYPE_GRAPH_EXECUTE_ACCEL_TIME_CYCLE 3003

/**
 * @brief QnnProfile_EventType_t definition to get profile information
 *        that corresponds to execute the graph on the accelerator
 *        when client invokes QnnGraph_execute or QnnGraph_executeAsync.
 *        The value returned is time taken in microseconds
 *
 * @note graph execute accelerator time maybe available on both
 *       QNN_PROFILE_LEVEL_BASIC and QNN_PROFILE_LEVEL_DETAILED levels
 *
 * @note When QNN_PROFILE_LEVEL_DETAILED is used, this event can have
 *       multiple sub-events of type QNN_PROFILE_EVENTTYPE_NODE / QNN_PROFILE_EVENTUNIT_MICROSEC.
 *       There will be a sub-event for each node that was added to the graph
 */
#define QNN_HTA_PROFILE_EVENTTYPE_GRAPH_EXECUTE_ACCEL_TIME_MICROSEC 3004

/**
 * @brief QnnProfile_EventType_t definition to get profile information
 *        that corresponds to time taken for miscellaneous work i.e. time
 *        that cannot be attributed to a node but are still needed to
 *        execute the graph on the accelerator. This occurs when client invokes
 *        QnnGraph_execute or QnnGraph_executeAsync.
 *        The value returned is time taken in microseconds
 *
 * @note graph execute misc accelerator time is available only on
 *       QNN_PROFILE_LEVEL_DETAILED levels
 */
#define QNN_HTA_PROFILE_EVENTTYPE_GRAPH_EXECUTE_MISC_ACCEL_TIME_MICROSEC 3005

/**
 * @brief QnnProfile_EventType_t definition to get profile information
 *        that corresponds to the remote procedure call on the ARM processor
 *        when client invokes QnnContext_free which in consequence deinit graph.
 *        The value returned is time in microseconds.
 *
 * @note graph deinit host time maybe available on both
 *       QNN_PROFILE_LEVEL_BASIC and QNN_PROFILE_LEVEL_DETAILED levels
 */
#define QNN_HTA_PROFILE_EVENTTYPE_GRAPH_DEINIT_HOST_TIME_MICROSEC 4001

/**
 * @brief QnnProfile_EventType_t definition to get profile information
 *        that corresponds to the remote procedure call on the HTA processor
 *        when client invokes QnnContext_free which in consequence deinit graph.
 *        The value returned is time in microseconds.
 *
 * @note graph deinit HTA time maybe available on both
 *       QNN_PROFILE_LEVEL_BASIC and QNN_PROFILE_LEVEL_DETAILED levels
 */
#define QNN_HTA_PROFILE_EVENTTYPE_GRAPH_DEINIT_HTA_TIME_MICROSEC 4002

/**
 * @brief QnnProfile_EventType_t definition to get profile information
 *        that corresponds to the time taken to deinit graph on the
 *        accelerator when client invokes QnnContext_free which in consequence
 *        deinit graph. The value returned is time in microseconds.
 *
 * @note graph deinit accelerator time maybe available on both
 *       QNN_PROFILE_LEVEL_BASIC and QNN_PROFILE_LEVEL_DETAILED levels
 */
#define QNN_HTA_PROFILE_EVENTTYPE_GRAPH_DEINIT_ACCEL_TIME_MICROSEC 4003

#ifdef __cplusplus
}
#endif

#endif  // QNN_HTA_PROFILE_H
