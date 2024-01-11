/* Documentation snippet
(C) 2017-2023 Niall Douglas <http://www.nedproductions.biz/> (3 commits), Luke Peterson <hazelnusse@gmail.com> (2 commits), Andrzej Krzemienski <akrzemi1@gmail.com> (2 commits) and Andrzej Krzemieński <akrzemi1@gmail.com> (1 commit)
File Created: Mar 2017


Boost Software License - Version 1.0 - August 17th, 2003

Permission is hereby granted, free of charge, to any person or organization
obtaining a copy of the software and accompanying documentation covered by
this license (the "Software") to use, reproduce, display, distribute,
execute, and transmit the Software, and to prepare derivative works of the
Software, and to permit third-parties to whom the Software is furnished to
do so, all subject to the following:

The copyright notices in the Software and this entire statement, including
the above license grant, this restriction and the following disclaimer,
must be included in all copies of the Software, in whole or in part, and
all derivative works of the Software, unless such copies or derivative
works are solely in the form of machine-executable object code generated by
a source language processor.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
*/

//! [error_code_registration]
#include <boost/system/error_code.hpp>  // bring in boost::system::error_code et al
#include <iostream>
#include <string>  // for string printing

// This is the custom error code enum
enum class ConversionErrc
{
  Success = 0,  // 0 should not represent an error
  EmptyString = 1,
  IllegalChar = 2,
  TooLong = 3,
};

namespace boost
{
  namespace system
  {
    // Tell the C++ 11 STL metaprogramming that enum ConversionErrc
    // is registered with the standard error code system
    template <> struct is_error_code_enum<ConversionErrc> : std::true_type
    {
    };
  }  // namespace system
}  // namespace boost

namespace detail
{
  // Define a custom error code category derived from boost::system::error_category
  class ConversionErrc_category : public boost::system::error_category
  {
  public:
    // Return a short descriptive name for the category
    virtual const char *name() const noexcept override final { return "ConversionError"; }
    // Return what each enum means in text
    virtual std::string message(int c) const override final
    {
      switch(static_cast<ConversionErrc>(c))
      {
      case ConversionErrc::Success:
        return "conversion successful";
      case ConversionErrc::EmptyString:
        return "converting empty string";
      case ConversionErrc::IllegalChar:
        return "got non-digit char when converting to a number";
      case ConversionErrc::TooLong:
        return "the number would not fit into memory";
      default:
        return "unknown";
      }
    }
    // OPTIONAL: Allow generic error conditions to be compared to me
    virtual boost::system::error_condition default_error_condition(int c) const noexcept override final
    {
      switch(static_cast<ConversionErrc>(c))
      {
      case ConversionErrc::EmptyString:
        return make_error_condition(boost::system::errc::invalid_argument);
      case ConversionErrc::IllegalChar:
        return make_error_condition(boost::system::errc::invalid_argument);
      case ConversionErrc::TooLong:
        return make_error_condition(boost::system::errc::result_out_of_range);
      default:
        // I have no mapping for this code
        return boost::system::error_condition(c, *this);
      }
    }
  };
}  // namespace detail

// Define the linkage for this function to be used by external code.
// This would be the usual __declspec(dllexport) or __declspec(dllimport)
// if we were in a Windows DLL etc. But for this example use a global
// instance but with inline linkage so multiple definitions do not collide.
#define THIS_MODULE_API_DECL extern inline

// Declare a global function returning a static instance of the custom category
THIS_MODULE_API_DECL const detail::ConversionErrc_category &ConversionErrc_category()
{
  static detail::ConversionErrc_category c;
  return c;
}


// Overload the global make_error_code() free function with our
// custom enum. It will be found via ADL by the compiler if needed.
inline boost::system::error_code make_error_code(ConversionErrc e)
{
  return {static_cast<int>(e), ConversionErrc_category()};
}

int main(void)
{
  // Note that we can now supply ConversionErrc directly to error_code
  boost::system::error_code ec = ConversionErrc::IllegalChar;

  std::cout << "ConversionErrc::IllegalChar is printed by boost::system::error_code as "
    << ec << " with explanatory message " << ec.message() << std::endl;

  // We can compare ConversionErrc containing error codes to generic conditions
  std::cout << "ec is equivalent to boost::system::errc::invalid_argument = "
    << (ec == std::errc::invalid_argument) << std::endl;
  std::cout << "ec is equivalent to boost::system::errc::result_out_of_range = "
    << (ec == std::errc::result_out_of_range) << std::endl;
  return 0;
}
//! [error_code_registration]
